"""Low-level seven-joint torque control for a dynamically enabled CoppeliaSim iiwa.

The visual-servo code remains the high-level planner.  This module is the
low-level actuator layer: it reads joint position/velocity, computes one
torque per iiwa joint, and advances CoppeliaSim in synchronous dynamics
steps.  It deliberately does not call ``sim.setJointPosition`` while active.

The controller uses joint-space PD with a slowly learned gravity/friction
bias.  CoppeliaSim does not expose a general inverse-dynamics API through the
remote API, so the bias is estimated only while the arm is nearly stationary;
it is a conservative practical gravity-compensation approximation rather than
a claimed exact M(q), C(q,dq), g(q) model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Any, Sequence

import numpy as np

from robot_state import (
    apply_shape_state,
    load_compatible as load_robot_state,
    zero_robot_velocities,
)


def _joint_mode(sim: Any, joint: int) -> int:
    raw = sim.getJointMode(joint)
    return int(raw[0] if isinstance(raw, (tuple, list)) else raw)


def _set_joint_mode(sim: Any, joint: int, mode: int) -> None:
    try:
        sim.setJointMode(joint, mode)
    except TypeError:
        sim.setJointMode(joint, mode, 0)


@dataclass(frozen=True)
class TorqueControllerConfig:
    """Conservative defaults for the stock LBR iiwa 7 R800 scene.

    ``kp``/``kd`` are used only by the explicit force/torque controller.  The
    dynamic ``position`` mode delegates joint position tracking to
    CoppeliaSim's dynamic position actuator, so those gains do not affect
    position-mode arm motion.
    """

    kp: tuple[float, ...] = (55.0, 55.0, 45.0, 35.0, 24.0, 16.0, 10.0)
    kd: tuple[float, ...] = (9.0, 9.0, 7.0, 5.5, 3.5, 2.2, 1.4)
    torque_limits: tuple[float, ...] = (45.0, 45.0, 35.0, 30.0, 22.0, 15.0, 10.0)
    bias_learning_rate: float = 0.015
    bias_velocity_limit: float = 0.08
    bias_error_limit: float = 0.035
    # Limit each IK reference update.  In dynamic position mode this is the
    # trajectory-rate limiter; kp/kd below are intentionally not used by the
    # position actuator (they only belong to the torque/force controller).
    max_target_step_rad: float = 0.02
    # Separate limits for CoppeliaSim's internal position actuator.  The
    # torque-mode limits above are intentionally conservative and are not
    # sufficient to hold the full iiwa chain in position mode.
    position_force_limits: tuple[float, ...] = (120.0, 120.0, 100.0, 80.0, 60.0, 40.0, 25.0)
    position_target_velocity_rad_s: float = 0.5
    # CoppeliaSim's stock iiwa has a very weak MuJoCo position PID.  A custom
    # joint callback is still a dynamic actuator: it returns bounded motor
    # velocity/effort at every physics substep while MuJoCo resolves gravity,
    # inertia and contact.  It is more predictable than changing scene PID
    # properties globally.
    position_callback_enabled: bool = False
    position_callback_gain: float = 20.0
    position_callback_damping: float = 0.35
    position_callback_velocity_limit_rad_s: float = 0.30
    velocity_position_gain: float = 3.0
    velocity_limit_rad_s: float = 0.45
    substeps_per_observation: int = 13
    startup_hold_steps: int = 30
    startup_error_limit_rad: float = 0.08
    torque_calibration_steps: int = 0
    # MuJoCo reports constraint-induced joint velocity oscillations for the
    # stock RG2 chain even when the TCP is visually stationary.  Use a short
    # consecutive window before contact; a separate pose-alignment gate below
    # still rejects any material TCP displacement.
    position_settle_velocity_limit_rad_s: float = 0.12
    position_settle_required_steps: int = 5
    position_settle_max_steps: int = 100

    @classmethod
    def from_environment(cls) -> "TorqueControllerConfig":
        import os

        def vector(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
            raw = os.environ.get(name)
            if not raw:
                return default
            values = tuple(float(item.strip()) for item in raw.split(","))
            if len(values) != 7:
                raise ValueError(f"{name} must contain exactly 7 comma-separated values")
            return values

        return cls(
            kp=vector("ROBOT_GRASP_TORQUE_KP", cls.kp),
            kd=vector("ROBOT_GRASP_TORQUE_KD", cls.kd),
            torque_limits=vector("ROBOT_GRASP_TORQUE_LIMITS", cls.torque_limits),
            position_force_limits=vector(
                "ROBOT_GRASP_POSITION_FORCE_LIMITS", cls.position_force_limits
            ),
            position_target_velocity_rad_s=float(
                os.environ.get(
                    "ROBOT_GRASP_POSITION_TARGET_VELOCITY_RAD_S",
                    cls.position_target_velocity_rad_s,
                )
            ),
            position_callback_enabled=os.environ.get(
                "ROBOT_GRASP_POSITION_CALLBACK_ENABLED",
                "1" if cls.position_callback_enabled else "0",
            ).lower()
            not in {"0", "false", "no"},
            position_callback_gain=float(
                os.environ.get(
                    "ROBOT_GRASP_POSITION_CALLBACK_GAIN",
                    cls.position_callback_gain,
                )
            ),
            position_callback_damping=float(
                os.environ.get(
                    "ROBOT_GRASP_POSITION_CALLBACK_DAMPING",
                    cls.position_callback_damping,
                )
            ),
            position_callback_velocity_limit_rad_s=float(
                os.environ.get(
                    "ROBOT_GRASP_POSITION_CALLBACK_VELOCITY_LIMIT_RAD_S",
                    cls.position_callback_velocity_limit_rad_s,
                )
            ),
            velocity_position_gain=float(
                os.environ.get(
                    "ROBOT_GRASP_VELOCITY_POSITION_GAIN",
                    cls.velocity_position_gain,
                )
            ),
            velocity_limit_rad_s=float(
                os.environ.get(
                    "ROBOT_GRASP_VELOCITY_LIMIT_RAD_S",
                    cls.velocity_limit_rad_s,
                )
            ),
            bias_learning_rate=float(os.environ.get("ROBOT_GRASP_TORQUE_BIAS_LR", cls.bias_learning_rate)),
            bias_velocity_limit=float(os.environ.get("ROBOT_GRASP_TORQUE_BIAS_VEL_LIMIT", cls.bias_velocity_limit)),
            bias_error_limit=float(os.environ.get("ROBOT_GRASP_TORQUE_BIAS_ERR_LIMIT", cls.bias_error_limit)),
            max_target_step_rad=float(os.environ.get("ROBOT_GRASP_TORQUE_MAX_TARGET_STEP_RAD", cls.max_target_step_rad)),
            substeps_per_observation=max(1, int(os.environ.get("ROBOT_GRASP_TORQUE_SUBSTEPS", cls.substeps_per_observation))),
            startup_hold_steps=max(0, int(os.environ.get("ROBOT_GRASP_DYNAMIC_STARTUP_HOLD_STEPS", cls.startup_hold_steps))),
            startup_error_limit_rad=float(os.environ.get("ROBOT_GRASP_DYNAMIC_STARTUP_ERROR_LIMIT_RAD", cls.startup_error_limit_rad)),
            torque_calibration_steps=max(
                0,
                int(
                    os.environ.get(
                        "ROBOT_GRASP_TORQUE_CALIBRATION_STEPS",
                        cls.torque_calibration_steps,
                    )
                ),
            ),
            position_settle_velocity_limit_rad_s=float(
                os.environ.get(
                    "ROBOT_GRASP_DYNAMIC_POSITION_SETTLE_VELOCITY_RAD_S",
                    cls.position_settle_velocity_limit_rad_s,
                )
            ),
            position_settle_required_steps=max(
                1,
                int(
                    os.environ.get(
                        "ROBOT_GRASP_DYNAMIC_POSITION_SETTLE_REQUIRED_STEPS",
                        cls.position_settle_required_steps,
                    )
                ),
            ),
            position_settle_max_steps=max(
                1,
                int(
                    os.environ.get(
                        "ROBOT_GRASP_DYNAMIC_POSITION_SETTLE_MAX_STEPS",
                        cls.position_settle_max_steps,
                    )
                ),
            ),
        )


class JointTorqueController:
    """Synchronous dynamic controller for exactly seven iiwa revolute joints."""

    def __init__(
        self,
        client: Any,
        sim: Any,
        joints: Sequence[int],
        config: TorqueControllerConfig | None = None,
        control_mode: str = "torque",
    ) -> None:
        if len(joints) != 7:
            raise ValueError(f"JointTorqueController requires 7 joints, got {len(joints)}")
        self.client = client
        self.sim = sim
        self.joints = tuple(int(joint) for joint in joints)
        self.config = config or TorqueControllerConfig.from_environment()
        self.control_mode = str(control_mode).strip().lower()
        if self.control_mode not in {"torque", "position", "velocity"}:
            raise ValueError("control_mode must be 'torque', 'position', or 'velocity'")
        self.original_modes = tuple(_joint_mode(sim, joint) for joint in self.joints)
        self.original_dynctrl: tuple[int | None, ...] = tuple(
            self._read_dynctrl_mode(joint) for joint in self.joints
        )
        self.original_script_enabled: int | None = None
        self.original_script_disabled: bool | None = None
        self.original_script_text: str | None = None
        self.position_callback_script_handle: int | None = None
        self.iiwa_script: int | None = None
        self.qd = self.read_positions()
        self.bias = np.zeros(7, dtype=np.float64)
        self.gravity_bias = np.zeros(7, dtype=np.float64)
        self.last_torque = np.zeros(7, dtype=np.float64)
        self.dynamic_dt = float(sim.getFloatParam(sim.floatparam_dynamic_step_size))
        if not math.isfinite(self.dynamic_dt) or self.dynamic_dt <= 0.0:
            self.dynamic_dt = 0.005
        try:
            self.simulation_dt = float(
                sim.getFloatParam(sim.floatparam_simulation_time_step)
            )
        except Exception:
            self.simulation_dt = 10.0 * self.dynamic_dt
        if not math.isfinite(self.simulation_dt) or self.simulation_dt <= 0.0:
            self.simulation_dt = 10.0 * self.dynamic_dt
        self.active = False
        self.started_simulation = False
        self._iiwa_handle: int | None = None
        self._original_model_property: int | None = None
        self._original_scripts_inactive = False
        self.position_callback_active = False
        self.startup_calibration: dict[str, Any] | None = None
        # This manifest is produced before physics-scene freezing.  It is the
        # source of truth for which links are truly dynamic; alias heuristics
        # are unsafe because RG2 has several repeated aliases (leftLink,
        # rightLink) and auxiliary mass shapes that are intentionally static.
        try:
            robot_base_for_manifest = int(self.sim.getObject("/iiwa"))
        except Exception:
            robot_base_for_manifest = int(self.sim.getObjectParent(self.joints[0]))
        self.robot_state_manifest = load_robot_state(self.sim, robot_base_for_manifest)

    def _read_dynctrl_mode(self, joint: int) -> int | None:
        parameter = getattr(self.sim, "jointintparam_dynctrlmode", None)
        if parameter is None:
            return None
        try:
            return int(self.sim.getObjectInt32Param(joint, parameter))
        except Exception:
            return None

    def read_positions(self) -> np.ndarray:
        return np.asarray([float(self.sim.getJointPosition(joint)) for joint in self.joints], dtype=np.float64)

    def read_velocities(self) -> np.ndarray:
        values: list[float] = []
        for joint in self.joints:
            try:
                values.append(float(self.sim.getJointVelocity(joint)))
            except Exception:
                values.append(0.0)
        return np.asarray(values, dtype=np.float64)

    def _zero_descendant_velocities(self) -> None:
        """Remove residual motion before a kinematic-to-dynamic hand-off."""
        setter = getattr(self.sim, "setObjectVelocity", None)
        if setter is None:
            return
        try:
            base = int(self.sim.getObject("/iiwa"))
        except Exception:
            base = int(self.sim.getObjectParent(self.joints[0]))
        zero_robot_velocities(self.sim, base)

    def _enable_dynamic_robot_shapes(self) -> None:
        """Re-enable the saved dynamic collision proxies at hand-off time.

        Scene generation freezes only the dynamic/respondable robot shapes so
        RGB-D and active perception can safely advance one refresh tick.  The
        visual meshes remain static/non-respondable.  At this point the arm
        controller is ready, so restore dynamic state for proxy links and the
        RG2 root/finger bodies.
        """
        static_param = getattr(self.sim, "shapeintparam_static", None)
        if static_param is None:
            return
        if self.robot_state_manifest is not None:
            # Restore exactly the flags captured from the tuned scene.  In
            # particular, this keeps centerAuxMass/leftAuxMass/rightAuxMass
            # static when the scene intentionally uses them as non-respondable
            # mass proxies, while re-enabling every saved dynamic body.
            apply_shape_state(
                self.sim,
                self.robot_state_manifest,
                freeze=False,
                reset_dynamic=True,
            )
            return
        try:
            base = int(self.sim.getObject("/iiwa"))
        except Exception:
            base = int(self.sim.getObjectParent(self.joints[0]))
        for handle in self.sim.getObjectsInTree(base, self.sim.sceneobject_shape, 0):
            try:
                alias = str(self.sim.getObjectAlias(handle)).lower().split("#", 1)[0]
                # Dynamic proxy convention used by the tuned iiwa/RG2 scene.
                if not (
                    alias.endswith("_resp")
                    or alias in {"rg2", "upperbase", "leftlink", "rightlink", "lefttouch", "righttouch", "leftauxmass", "rightauxmass", "centerauxmass"}
                ):
                    continue
                self.sim.setObjectInt32Param(handle, static_param, 0)
                reset_dynamic = getattr(self.sim, "resetDynamicObject", None)
                if reset_dynamic is not None:
                    try:
                        reset_dynamic(int(handle))
                    except Exception:
                        pass
            except Exception:
                pass

    def configure_dynamic_mode(self) -> None:
        """Put every iiwa joint in dynamic position or force/torque mode."""
        # Estimate the hold torque while the chain is still kinematic.  Calling
        # setJointPosition on a dynamic joint is a state override and can
        # inject artificial velocity/energy into the physics solver.
        initial_q = self.read_positions()
        dynamic_mode = getattr(self.sim, "jointmode_dynamic", getattr(self.sim, "jointmode_force", 5))
        calibrate_torque = bool(
            self.control_mode == "torque" and self.config.torque_calibration_steps > 0
        )
        use_position_callback = bool(
            self.control_mode == "position"
            and self.config.position_callback_enabled
            and hasattr(self.sim, "jointdynctrl_callback")
        )
        if use_position_callback:
            dynctrl_mode = getattr(self.sim, "jointdynctrl_callback")
        elif self.control_mode == "position" or calibrate_torque:
            dynctrl_mode = getattr(self.sim, "jointdynctrl_position", 8)
        elif self.control_mode == "velocity":
            dynctrl_mode = getattr(self.sim, "jointdynctrl_velocity", 4)
        else:
            dynctrl_mode = getattr(self.sim, "jointdynctrl_force", 1)
        dynctrl_param = getattr(self.sim, "jointintparam_dynctrlmode", None)
        # The scene may have frozen the dynamic proxy links while perception
        # was running.  Restore their saved dynamic flags before estimating
        # gravity; otherwise the estimator would see zero mass and the torque
        # fallback would have no compensation.
        self._enable_dynamic_robot_shapes()
        # Compute the gravity estimate before enabling dynamics.  Position
        # mode does not use it while its internal actuator is healthy, but it
        # is needed for the explicit torque fallback if that actuator cannot
        # hold this model's heavy link proxies.
        gravity_bias = self.estimate_gravity_torque(initial_q)
        for joint in self.joints:
            _set_joint_mode(self.sim, joint, dynamic_mode)
            if dynctrl_param is not None:
                self.sim.setObjectInt32Param(joint, dynctrl_param, dynctrl_mode)
            motor_param = getattr(self.sim, "jointintparam_motor_enabled", None)
            if motor_param is not None:
                try:
                    self.sim.setObjectInt32Param(joint, motor_param, 1)
                except Exception:
                    pass
        # The stock iiwa customization script contains only a demonstration
        # trajectory and must not compete with the external arm controller.
        # Do not disable the whole model: RG2's child script is part of that
        # model and must continue to receive open/close commands.
        try:
            iiwa = int(self.sim.getObject("/iiwa"))
            script = self.sim.getScriptAssociatedWithObject(iiwa)
            if script is not None and int(script) >= 0:
                self._iiwa_handle = iiwa
                self.iiwa_script = int(script)
                self._original_model_property = int(self.sim.getModelProperty(iiwa))
                if use_position_callback:
                    self.original_script_disabled = bool(
                        self.sim.getBoolProperty(int(script), "scriptDisabled")
                    )
                    self.sim.setBoolProperty(int(script), "scriptDisabled", True)
                    callback_script = int(
                        self.sim.createScript(
                            self.sim.scripttype_simulation,
                            self._position_callback_script(),
                            0,
                            "lua",
                        )
                    )
                    # Joint callbacks are discovered from scripts attached to
                    # a joint (or below one).  Attach to joint 1 so the script
                    # lies on the ancestor chain of all seven iiwa joints.
                    self.sim.setObjectParent(callback_script, self.joints[0], True)
                    try:
                        self.sim.setObjectAlias(
                            callback_script,
                            "iiwa_dynamic_joint_controller",
                        )
                    except Exception:
                        pass
                    self.position_callback_script_handle = callback_script
                    self.position_callback_active = True
                # Disable only the iiwa simulation script.  Current
                # CoppeliaSim releases expose this through the writable
                # ``scriptDisabled`` property; the legacy script int
                # parameter may return None and silently leave the demo
                # trajectory active, which then overwrites every external
                # torque/velocity command.  RG2 has its own script object and
                # remains enabled.
                if not use_position_callback:
                    try:
                        self.original_script_disabled = bool(
                            self.sim.getBoolProperty(int(script), "scriptDisabled")
                        )
                        self.sim.setBoolProperty(int(script), "scriptDisabled", True)
                        self._original_scripts_inactive = True
                    except Exception:
                        pass
                    enabled_param = getattr(self.sim, "scriptintparam_enabled", None)
                    if enabled_param is not None and not self._original_scripts_inactive:
                        try:
                            try:
                                raw_enabled = self.sim.getScriptInt32Param(
                                    int(script), enabled_param
                                )
                                if raw_enabled is not None:
                                    self.original_script_enabled = int(raw_enabled)
                            except Exception:
                                pass
                            self.sim.setScriptInt32Param(int(script), enabled_param, 0)
                            self._original_scripts_inactive = True
                        except Exception:
                            pass
        except Exception:
            pass
        self.qd = initial_q
        self.gravity_bias = gravity_bias
        self.bias = np.zeros(7, dtype=np.float64)
        self._zero_descendant_velocities()
        # Prime the actuator before starting dynamics.  Otherwise the first
        # physics tick can see a dynamic chain with no target/force command.
        if self.control_mode == "position" or calibrate_torque:
            limits = np.asarray(self.config.position_force_limits, dtype=np.float64)
            target_velocity = float(self.config.position_target_velocity_rad_s)
            if not math.isfinite(target_velocity) or target_velocity <= 0.0:
                raise ValueError("position target velocity must be positive and finite")
            for joint, position, force_limit in zip(self.joints, self.qd, limits):
                try:
                    self.sim.setJointTargetPosition(int(joint), float(position))
                    self.sim.setJointTargetVelocity(int(joint), target_velocity)
                    self.sim.setJointTargetForce(int(joint), float(abs(force_limit)), True)
                except TypeError:
                    try:
                        self.sim.setJointTargetForce(int(joint), float(abs(force_limit)))
                    except Exception:
                        pass
                except Exception:
                    pass
        elif self.control_mode == "velocity":
            limits = np.asarray(self.config.position_force_limits, dtype=np.float64)
            for joint, force_limit in zip(self.joints, limits):
                self.sim.setJointTargetVelocity(int(joint), 0.0)
                try:
                    self.sim.setJointTargetForce(int(joint), float(abs(force_limit)), True)
                except TypeError:
                    self.sim.setJointTargetForce(int(joint), float(abs(force_limit)))
        self.active = True

    def _position_callback_script(self) -> str:
        """Build the temporary low-level iiwa controller for custom mode."""

        gain = float(self.config.position_callback_gain)
        damping = float(self.config.position_callback_damping)
        velocity_limit = float(self.config.position_callback_velocity_limit_rad_s)
        if not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("position callback gain must be positive and finite")
        if not math.isfinite(damping) or damping < 0.0:
            raise ValueError("position callback damping must be finite and non-negative")
        if not math.isfinite(velocity_limit) or velocity_limit <= 0.0:
            raise ValueError("position callback velocity limit must be positive and finite")
        handles = ", ".join(f"[{int(joint)}]=true" for joint in self.joints)
        return f"""sim=require'sim'

local controlled={{{handles}}}
local gain={gain:.12g}
local damping={damping:.12g}
local velocityLimit={velocity_limit:.12g}

function sysCall_joint(inData)
    if not controlled[inData.handle] then
        return
    end
    local velocity=inData.error*gain
    if inData.vel then
        velocity=velocity-inData.vel*damping
    end
    local limit=velocityLimit
    if inData.maxVel and inData.maxVel>0 and inData.maxVel<limit then
        limit=inData.maxVel
    end
    if velocity>limit then velocity=limit end
    if velocity<-limit then velocity=-limit end
    local effort=inData.maxForce
    if not effort then effort=inData.force end
    if not effort then effort=0 end
    if effort<0 then effort=-effort end
    return {{vel=velocity, force=effort}}
end
"""

    def switch_position_to_callback(self) -> dict[str, Any]:
        """Switch a position-controlled arm to bounded custom control.

        The built-in MuJoCo position actuator is used for the long hand-off and
        RG2 close because it provides a quiet hold.  Its stock PID has a large
        gravity-loaded steady-state error and cannot execute the short lift.
        Once bilateral contact is established, this callback supplies the
        standard error-to-motor-velocity law at each dynamics substep.
        """

        if not self.active or self.control_mode != "position":
            raise RuntimeError(
                "Position callback switch requires an active position controller"
            )
        if self.position_callback_active:
            return {
                "active": True,
                "script_handle": self.position_callback_script_handle,
            }
        callback_mode = getattr(self.sim, "jointdynctrl_callback", None)
        dynctrl_param = getattr(self.sim, "jointintparam_dynctrlmode", None)
        if callback_mode is None or dynctrl_param is None:
            raise RuntimeError("CoppeliaSim custom joint callback mode is unavailable")
        # The caller owns synchronous stepping.  Remote calls made after one
        # client.step() returns and before the next one execute at the same
        # simulation boundary, so no uncontrolled physics tick occurs here.
        # Hold the measured joints first, install the callback, and only then
        # switch all seven actuator modes.  Roll back the partial transaction
        # if any API call fails.
        held_q = self.read_positions()
        previous_modes = [self._read_dynctrl_mode(joint) for joint in self.joints]
        callback_script: int | None = None
        self.qd = held_q.copy()
        self.apply_position_target()
        try:
            callback_script = int(
                self.sim.createScript(
                    self.sim.scripttype_simulation,
                    self._position_callback_script(),
                    0,
                    "lua",
                )
            )
            self.sim.setObjectParent(callback_script, self.joints[0], True)
            try:
                self.sim.setObjectAlias(
                    callback_script,
                    "iiwa_dynamic_joint_controller",
                )
            except Exception:
                pass
            for joint in self.joints:
                self.sim.setObjectInt32Param(
                    joint,
                    dynctrl_param,
                    int(callback_mode),
                )
            self.position_callback_script_handle = callback_script
            self.position_callback_active = True
            self.apply_position_target()
        except Exception:
            for joint, mode in zip(self.joints, previous_modes):
                if mode is None:
                    continue
                try:
                    self.sim.setObjectInt32Param(joint, dynctrl_param, int(mode))
                except Exception:
                    pass
            if callback_script is not None:
                try:
                    self.sim.removeObject(callback_script)
                except Exception:
                    pass
            self.position_callback_script_handle = None
            self.position_callback_active = False
            self.apply_position_target()
            raise
        return {
            "active": True,
            "script_handle": callback_script,
            "gain": float(self.config.position_callback_gain),
            "damping": float(self.config.position_callback_damping),
            "velocity_limit_rad_s": float(
                self.config.position_callback_velocity_limit_rad_s
            ),
        }

    def _calibrate_torque_hold(self, initial_q: np.ndarray) -> dict[str, Any]:
        """Measure the actual seven-joint hold effort before force control.

        The potential-energy estimate cannot represent all constrained RG2
        loads in a CoppeliaSim scene.  A short dynamic position hold exposes
        the signed motor torques that MuJoCo actually needs at the hand-off
        pose.  Those measurements become the feed-forward term for the
        subsequent external force/torque controller; no joint state is
        overwritten while dynamics is running.
        """

        count = max(1, int(self.config.torque_calibration_steps))
        limits = np.asarray(self.config.position_force_limits, dtype=np.float64)
        target_velocity = float(self.config.position_target_velocity_rad_s)
        samples: list[np.ndarray] = []
        max_error = 0.0
        max_velocity = 0.0
        for index in range(count):
            for joint, position, force_limit in zip(self.joints, initial_q, limits):
                self.sim.setJointTargetPosition(int(joint), float(position))
                self.sim.setJointTargetVelocity(int(joint), target_velocity)
                try:
                    self.sim.setJointTargetForce(int(joint), float(abs(force_limit)), False)
                except TypeError:
                    self.sim.setJointTargetForce(int(joint), float(abs(force_limit)))
            self.client.step()
            q = self.read_positions()
            dq = self.read_velocities()
            max_error = max(max_error, float(np.max(np.abs(q - initial_q))))
            max_velocity = max(max_velocity, float(np.max(np.abs(dq))))
            if index >= count // 2:
                measured: list[float] = []
                for joint in self.joints:
                    try:
                        measured.append(float(self.sim.getJointForce(int(joint))))
                    except Exception:
                        measured.append(float("nan"))
                values = np.asarray(measured, dtype=np.float64)
                if np.all(np.isfinite(values)):
                    samples.append(values)
        if not samples:
            measured_hold = self.gravity_bias.copy()
            source = "potential_fallback"
        else:
            measured_hold = np.median(np.asarray(samples, dtype=np.float64), axis=0)
            source = "mujoco_position_hold"
        measured_hold = np.clip(
            measured_hold,
            -np.asarray(self.config.torque_limits, dtype=np.float64),
            np.asarray(self.config.torque_limits, dtype=np.float64),
        )
        dynctrl_param = getattr(self.sim, "jointintparam_dynctrlmode", None)
        force_mode = getattr(self.sim, "jointdynctrl_force", 1)
        if dynctrl_param is not None:
            for joint in self.joints:
                self.sim.setObjectInt32Param(int(joint), dynctrl_param, force_mode)
        self.gravity_bias = measured_hold
        self.bias = np.zeros(7, dtype=np.float64)
        self.qd = initial_q.copy()
        torque, q, dq = self.apply_torque()
        return {
            "source": source,
            "steps": count,
            "max_position_error_rad": max_error,
            "max_velocity_rad_s": max_velocity,
            "measured_hold_torque": measured_hold.tolist(),
            "first_force_command": torque.tolist(),
            "q": q.tolist(),
            "dq": dq.tolist(),
        }

    def switch_position_to_torque_hold(self) -> dict[str, Any]:
        """Switch a running position reference to a gravity-compensated hold.

        Some stock iiwa scenes expose a dynamic position actuator that does
        not supply enough stable gravity compensation for the proxy-link
        masses.  If the position reference fails its settling gate, keeping
        it active would let the arm collapse.  This explicit fallback changes
        only the seven arm joints to force mode, freezes the *measured* pose
        as the new reference, and lets the existing PD+gravity controller
        arrest the motion.
        """
        if not self.active or self.control_mode != "position":
            raise RuntimeError("Position-to-torque fallback requires active position mode")
        self.control_mode = "torque"
        dynctrl_param = getattr(self.sim, "jointintparam_dynctrlmode", None)
        force_mode = getattr(self.sim, "jointdynctrl_force", 1)
        if dynctrl_param is not None:
            for joint in self.joints:
                try:
                    self.sim.setObjectInt32Param(joint, dynctrl_param, force_mode)
                except Exception:
                    pass
        self.qd = self.read_positions()
        # Apply immediately before the next synchronous step.  No joint
        # position write is performed while the simulation is dynamic.
        torque, q, dq = self.apply_torque()
        return {
            "control_mode": "torque",
            "fallback": True,
            "q": q.tolist(),
            "dq": dq.tolist(),
            "torque": torque.tolist(),
        }

    def configure_dynamic_force_mode(self) -> None:
        """Backward-compatible alias for the torque mode."""
        self.control_mode = "torque"
        self.configure_dynamic_mode()

    def estimate_gravity_torque(self, q: Sequence[float]) -> np.ndarray:
        """Estimate g(q) from the scene's dynamic shape masses and COMs.

        CoppeliaSim's public remote API does not provide a general robot
        inverse-dynamics call.  At a stopped, kinematic configuration we can
        nevertheless evaluate U(q)=sum(m*g*z_com) and finite-difference its
        gradient.  The scene is restored exactly before returning.
        """
        q0 = np.asarray(q, dtype=np.float64).reshape(7)
        static_param = getattr(self.sim, "shapeintparam_static", None)
        shapes: list[tuple[int, float]] = []
        try:
            base = int(self.sim.getObjectParent(self.joints[0]))
            for handle in self.sim.getObjectsInTree(base, self.sim.sceneobject_shape, 0):
                if static_param is not None:
                    try:
                        if int(self.sim.getObjectInt32Param(handle, static_param)) != 0:
                            continue
                    except Exception:
                        pass
                try:
                    mass = float(self.sim.getShapeMass(handle))
                except Exception:
                    continue
                if math.isfinite(mass) and mass > 0.0:
                    shapes.append((int(handle), mass))
            if not shapes:
                return np.zeros(7, dtype=np.float64)
            gravity = np.asarray(self.sim.getArrayParam(self.sim.arrayparam_gravity), dtype=np.float64)
            if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
                gravity = np.asarray([0.0, 0.0, -9.81], dtype=np.float64)
            potential = lambda: float(sum(
                mass * float(np.dot(-gravity, np.asarray(self.sim.getObjectPosition(handle, -1), dtype=np.float64)))
                for handle, mass in shapes
            ))
            delta = 1e-4
            result = np.zeros(7, dtype=np.float64)
            for index in range(7):
                plus = q0.copy(); plus[index] += delta
                minus = q0.copy(); minus[index] -= delta
                for joint, value in zip(self.joints, plus):
                    self.sim.setJointPosition(joint, float(value))
                u_plus = potential()
                for joint, value in zip(self.joints, minus):
                    self.sim.setJointPosition(joint, float(value))
                u_minus = potential()
                # A static hold requires the actuator torque to cancel the
                # generalized gravitational force.  With U(q)=m*g*z_com,
                # that compensation is +dU/dq (gravity itself contributes
                # -dU/dq in the equation of motion).
                result[index] = (u_plus - u_minus) / (2.0 * delta)
            for joint, value in zip(self.joints, q0):
                self.sim.setJointPosition(joint, float(value))
            limits = np.asarray(self.config.torque_limits, dtype=np.float64)
            return np.clip(result, -limits, limits)
        except Exception:
            for joint, value in zip(self.joints, q0):
                try:
                    self.sim.setJointPosition(joint, float(value))
                except Exception:
                    pass
            return np.zeros(7, dtype=np.float64)

    def set_target(self, qd: Sequence[float]) -> np.ndarray:
        requested = np.asarray(qd, dtype=np.float64).reshape(7)
        delta = requested - self.qd
        max_step = max(float(self.config.max_target_step_rad), 1e-6)
        delta = np.clip(delta, -max_step, max_step)
        self.qd = self.qd + delta
        return self.qd.copy()

    def compute_torque(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self.read_positions()
        dq = self.read_velocities()
        error = self.qd - q
        pd = np.asarray(self.config.kp) * error - np.asarray(self.config.kd) * dq
        stationary = float(np.max(np.abs(dq))) <= self.config.bias_velocity_limit
        close = float(np.max(np.abs(error))) <= self.config.bias_error_limit
        if stationary and close:
            alpha = float(np.clip(self.config.bias_learning_rate, 0.0, 1.0))
            self.bias = (1.0 - alpha) * self.bias + alpha * pd
        raw = pd + self.gravity_bias + self.bias
        limits = np.asarray(self.config.torque_limits, dtype=np.float64)
        torque = np.clip(raw, -limits, limits)
        self.last_torque = torque
        return torque, q, dq

    def apply_torque(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.active:
            raise RuntimeError("Torque controller is not configured")
        torque, q, dq = self.compute_torque()
        for joint, value in zip(self.joints, torque):
            # The third Lua argument is signedValue=true.  In Python's ZMQ
            # API the signed target is represented by the signed scalar.
            self.sim.setJointTargetForce(joint, float(value), True)
        return torque, q, dq

    def apply_position_target(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.active:
            raise RuntimeError("Dynamic position controller is not configured")
        q = self.read_positions()
        dq = self.read_velocities()
        limits = np.asarray(self.config.position_force_limits, dtype=np.float64)
        target_velocity = float(self.config.position_target_velocity_rad_s)
        if not math.isfinite(target_velocity) or target_velocity <= 0.0:
            raise ValueError("position target velocity must be positive and finite")
        for joint, position, force_limit in zip(self.joints, self.qd, limits):
            # Position mode uses CoppeliaSim's dynamic position actuator.
            # TorqueControllerConfig.kp/kd intentionally do not participate.
            self.sim.setJointTargetPosition(joint, float(position))
            try:
                # In custom callback mode the callback computes the signed
                # motor velocity from the position error.  The target velocity
                # remains only a conservative per-joint ceiling/fallback.
                self.sim.setJointTargetVelocity(joint, target_velocity)
                self.sim.setJointTargetForce(joint, float(abs(force_limit)), True)
            except TypeError:
                self.sim.setJointTargetForce(joint, float(abs(force_limit)))
        self.last_torque = np.zeros(7, dtype=np.float64)
        return self.last_torque.copy(), q, dq

    def apply_velocity_target(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track ``qd`` through MuJoCo's force-limited velocity actuator.

        No joint position is written.  The outer position loop produces a
        bounded joint velocity, while CoppeliaSim/MuJoCo resolves motor force,
        gravity, contact and inertia at each dynamics step.
        """

        if not self.active or self.control_mode != "velocity":
            raise RuntimeError("Dynamic velocity controller is not configured")
        q = self.read_positions()
        dq = self.read_velocities()
        gain = float(self.config.velocity_position_gain)
        velocity_limit = float(self.config.velocity_limit_rad_s)
        if not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("velocity position gain must be positive and finite")
        if not math.isfinite(velocity_limit) or velocity_limit <= 0.0:
            raise ValueError("velocity limit must be positive and finite")
        target_velocity = np.clip(gain * (self.qd - q), -velocity_limit, velocity_limit)
        limits = np.asarray(self.config.position_force_limits, dtype=np.float64)
        for joint, velocity, force_limit in zip(self.joints, target_velocity, limits):
            self.sim.setJointTargetVelocity(int(joint), float(velocity))
            try:
                self.sim.setJointTargetForce(int(joint), float(abs(force_limit)), False)
            except TypeError:
                self.sim.setJointTargetForce(int(joint), float(abs(force_limit)))
        self.last_torque = np.zeros(7, dtype=np.float64)
        return self.last_torque.copy(), q, dq

    def settle_position_target(
        self,
        *,
        velocity_limit_rad_s: float | None = None,
        required_steps: int | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Wait for the dynamic position arm to settle at its current ``qd``.

        The reference is held fixed while each synchronous physics step first
        reapplies the position target and then advances the simulation.  The
        method is deliberately arm-only: it does not publish RG2 signals or
        change the RG2 child-script authority.  A failure raises before a
        physical close command can be issued.
        """
        if not self.active or self.control_mode not in {"position", "velocity"}:
            raise RuntimeError(
                "Dynamic settling requires an active position or velocity controller"
            )
        limit = float(
            self.config.position_settle_velocity_limit_rad_s
            if velocity_limit_rad_s is None
            else velocity_limit_rad_s
        )
        required = max(
            1,
            int(
                self.config.position_settle_required_steps
                if required_steps is None
                else required_steps
            ),
        )
        budget = max(
            required,
            int(
                self.config.position_settle_max_steps
                if max_steps is None
                else max_steps
            ),
        )
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("position settling velocity limit must be positive and finite")
        fixed_qd = self.qd.copy()
        consecutive = 0
        max_velocity_seen = 0.0
        for step_index in range(1, budget + 1):
            # Keep the current IK reference fixed.  In particular, do not call
            # set_target here, because that would introduce another reference
            # step while the arm is supposed to be settling.
            self.qd = fixed_qd.copy()
            if self.control_mode == "position":
                self.apply_position_target()
            else:
                self.apply_velocity_target()
            self.client.step()
            dq = self.read_velocities()
            max_velocity = float(np.max(np.abs(dq)))
            max_velocity_seen = max(max_velocity_seen, max_velocity)
            if max_velocity < limit:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= required:
                return {
                    "stable": True,
                    "steps": step_index,
                    "required_steps": required,
                    "max_velocity_rad_s": max_velocity,
                    "max_velocity_seen_rad_s": max_velocity_seen,
                    "velocity_limit_rad_s": limit,
                    "qd": fixed_qd.tolist(),
                }
        raise RuntimeError(
            "Dynamic position arm did not settle before physical RG2 close: "
            f"max_steps={budget}, required_consecutive={required}, "
            f"velocity_limit={limit:.5f} rad/s, "
            f"max_velocity_seen={max_velocity_seen:.5f} rad/s"
        )

    def step(self, qd: Sequence[float] | None = None, steps: int | None = None) -> dict[str, Any]:
        if qd is not None:
            self.set_target(qd)
        count = max(1, int(steps if steps is not None else self.config.substeps_per_observation))
        for _ in range(count):
            if self.control_mode == "position":
                torque, q, dq = self.apply_position_target()
            elif self.control_mode == "velocity":
                torque, q, dq = self.apply_velocity_target()
            else:
                torque, q, dq = self.apply_torque()
            self.client.step()
        return {
            "q": q.tolist(),
            "dq": dq.tolist(),
            "qd": self.qd.tolist(),
            "torque": torque.tolist(),
            "bias": self.bias.tolist(),
            "steps": count,
            "dt": self.dynamic_dt,
        }

    def hold_current(self, steps: int | None = None) -> dict[str, Any]:
        self.qd = self.read_positions()
        return self.step(self.qd, steps=steps)

    def start(self) -> None:
        try:
            if not self.active:
                self.configure_dynamic_mode()
            self.client.setStepping(True)
            state = self.sim.getSimulationState()
            if state in {self.sim.simulation_stopped, self.sim.simulation_paused}:
                # Targets were primed in configure_dynamic_mode.  Start only
                # after the command exists, eliminating the uncontrolled first step.
                self.sim.startSimulation()
                self.started_simulation = True
            if self.control_mode == "torque" and self.config.torque_calibration_steps > 0:
                self.startup_calibration = self._calibrate_torque_hold(self.qd.copy())
            # Establish a zero-error dynamic hand-off before IK/PBVS sends a new
            # target.  This prevents a paused kinematic pose from becoming a large
            # impulse when the seven arm joints are first made dynamic.
            hold_steps = int(self.config.startup_hold_steps)
            # The tuned stock scene's position actuator is known to lose the
            # gravity hold during its first few ticks.  When the explicit
            # safety fallback is enabled, do not spend 30 ticks allowing that
            # actuator to accelerate the arm before switching to torque mode.
            if (
                self.control_mode == "position"
                and os.environ.get(
                    "ROBOT_GRASP_DYNAMIC_POSITION_FALLBACK_TORQUE", "1"
                ).lower()
                not in {"0", "false", "no"}
            ):
                hold_steps = min(1, hold_steps)
            if self.control_mode in {"position", "velocity"} and hold_steps > 0:
                q0 = self.read_positions()
                self.qd = q0.copy()
                max_error = 0.0
                for _ in range(hold_steps):
                    result = self.step(q0, steps=1)
                    q = np.asarray(result["q"], dtype=np.float64)
                    max_error = max(max_error, float(np.max(np.abs(q - q0))))
                if max_error > float(self.config.startup_error_limit_rad):
                    raise RuntimeError(
                        "Dynamic arm hand-off is unstable: "
                        f"max startup position error={math.degrees(max_error):.2f} deg"
                    )
        except Exception:
            # A failed startup must not leave the scene running with dynamic
            # joints but without a valid owner.  The caller can then safely
            # report the hand-off failure and start a fresh scene.
            try:
                self.close()
            except Exception:
                pass
            raise

    def stop(self) -> None:
        try:
            if self.sim.getSimulationState() not in {self.sim.simulation_stopped}:
                self.sim.stopSimulation()
                deadline = time.monotonic() + 5.0
                while self.sim.getSimulationState() != self.sim.simulation_stopped:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("CoppeliaSim did not reach stopped state within 5 seconds")
                    time.sleep(0.01)
        finally:
            self.started_simulation = False

    def diagnostics(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "control_mode": self.control_mode,
            "position_callback_active": self.position_callback_active,
            "dynamic_dt_s": self.dynamic_dt,
            "simulation_dt_s": self.simulation_dt,
            "joint_modes": [_joint_mode(self.sim, joint) for joint in self.joints],
            "dynctrl_modes": [self._read_dynctrl_mode(joint) for joint in self.joints],
            "q": self.read_positions().tolist(),
            "dq": self.read_velocities().tolist(),
            "last_torque": self.last_torque.tolist(),
            "gravity_bias": self.bias.tolist(),
            "gravity_bias_estimate": self.gravity_bias.tolist(),
            "startup_calibration": self.startup_calibration,
        }

    def restore_joint_modes(self) -> None:
        for joint, mode, dynctrl in zip(self.joints, self.original_modes, self.original_dynctrl):
            try:
                _set_joint_mode(self.sim, joint, mode)
            except Exception:
                pass
            if dynctrl is not None:
                parameter = getattr(self.sim, "jointintparam_dynctrlmode", None)
                if parameter is not None:
                    try:
                        self.sim.setObjectInt32Param(joint, parameter, dynctrl)
                    except Exception:
                        pass
        if self._iiwa_handle is not None and self._original_model_property is not None:
            # Restore the model property only.  The per-script switch is
            # restored by CoppeliaSim when the stopped scene is restarted.
            try: self.sim.setModelProperty(self._iiwa_handle, self._original_model_property)
            except Exception: pass
        if self.position_callback_script_handle is not None:
            try:
                self.sim.removeObject(self.position_callback_script_handle)
            except Exception:
                pass
            self.position_callback_script_handle = None
            if self.iiwa_script is not None:
                try:
                    self.sim.setBoolProperty(
                        self.iiwa_script,
                        "scriptDisabled",
                        False
                        if self.original_script_disabled is None
                        else bool(self.original_script_disabled),
                    )
                except Exception:
                    pass
            self.position_callback_active = False
        if self.iiwa_script is not None and self._original_scripts_inactive:
            try:
                if self.original_script_disabled is not None:
                    self.sim.setBoolProperty(
                        self.iiwa_script,
                        "scriptDisabled",
                        bool(self.original_script_disabled),
                    )
                else:
                    enabled_param = getattr(self.sim, "scriptintparam_enabled", None)
                    if enabled_param is None:
                        raise RuntimeError("script enabled parameter unavailable")
                    self.sim.setScriptInt32Param(
                        self.iiwa_script,
                        enabled_param,
                        1 if self.original_script_enabled is None else self.original_script_enabled,
                    )
            except Exception:
                pass
        self.active = False

    def close(self) -> None:
        try:
            if self.started_simulation:
                self.stop()
        finally:
            # Do not leave the stopped scene with a half dynamic/half frozen
            # robot.  CoppeliaSim does not automatically undo shape flags
            # changed through the remote API on stop; restoring the exact
            # scene-tree baseline here makes the next run deterministic even
            # when the caller is not pipeline_runner.py.
            if self.robot_state_manifest is not None:
                try:
                    apply_shape_state(
                        self.sim,
                        self.robot_state_manifest,
                        freeze=False,
                        reset_dynamic=False,
                    )
                    base = int(self.sim.getObject("/iiwa"))
                    zero_robot_velocities(self.sim, base)
                except Exception:
                    pass
            self.restore_joint_modes()
            try:
                # Do not leak synchronous stepping state into the next run;
                # the next controller will explicitly enable it again.
                self.client.setStepping(False)
            except Exception:
                pass
