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
import time
from typing import Any, Sequence

import numpy as np


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
    """Conservative defaults for the stock LBR iiwa 7 R800 scene."""

    kp: tuple[float, ...] = (55.0, 55.0, 45.0, 35.0, 24.0, 16.0, 10.0)
    kd: tuple[float, ...] = (9.0, 9.0, 7.0, 5.5, 3.5, 2.2, 1.4)
    torque_limits: tuple[float, ...] = (45.0, 45.0, 35.0, 30.0, 22.0, 15.0, 10.0)
    bias_learning_rate: float = 0.015
    bias_velocity_limit: float = 0.08
    bias_error_limit: float = 0.035
    max_target_step_rad: float = 0.10
    substeps_per_observation: int = 13

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
            bias_learning_rate=float(os.environ.get("ROBOT_GRASP_TORQUE_BIAS_LR", cls.bias_learning_rate)),
            bias_velocity_limit=float(os.environ.get("ROBOT_GRASP_TORQUE_BIAS_VEL_LIMIT", cls.bias_velocity_limit)),
            bias_error_limit=float(os.environ.get("ROBOT_GRASP_TORQUE_BIAS_ERR_LIMIT", cls.bias_error_limit)),
            max_target_step_rad=float(os.environ.get("ROBOT_GRASP_TORQUE_MAX_TARGET_STEP_RAD", cls.max_target_step_rad)),
            substeps_per_observation=max(1, int(os.environ.get("ROBOT_GRASP_TORQUE_SUBSTEPS", cls.substeps_per_observation))),
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
        if self.control_mode not in {"torque", "position"}:
            raise ValueError("control_mode must be 'torque' or 'position'")
        self.original_modes = tuple(_joint_mode(sim, joint) for joint in self.joints)
        self.original_dynctrl: tuple[int | None, ...] = tuple(
            self._read_dynctrl_mode(joint) for joint in self.joints
        )
        self.original_script_enabled: int | None = None
        self.iiwa_script: int | None = None
        self.qd = self.read_positions()
        self.bias = np.zeros(7, dtype=np.float64)
        self.gravity_bias = np.zeros(7, dtype=np.float64)
        self.last_torque = np.zeros(7, dtype=np.float64)
        self.dynamic_dt = float(sim.getFloatParam(sim.floatparam_dynamic_step_size))
        if not math.isfinite(self.dynamic_dt) or self.dynamic_dt <= 0.0:
            self.dynamic_dt = 0.005
        self.active = False
        self.started_simulation = False
        self._iiwa_handle: int | None = None
        self._original_model_property: int | None = None
        self._original_scripts_inactive = False

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

    def configure_dynamic_mode(self) -> None:
        """Put every iiwa joint in dynamic position or force/torque mode."""
        # Estimate the hold torque while the chain is still kinematic.  Calling
        # setJointPosition on a dynamic joint is a state override and can
        # inject artificial velocity/energy into the physics solver.
        initial_q = self.read_positions()
        dynamic_mode = getattr(self.sim, "jointmode_dynamic", getattr(self.sim, "jointmode_force", 5))
        if self.control_mode == "position":
            dynctrl_mode = getattr(self.sim, "jointdynctrl_position", 8)
        else:
            dynctrl_mode = getattr(self.sim, "jointdynctrl_force", 1)
        dynctrl_param = getattr(self.sim, "jointintparam_dynctrlmode", None)
        gravity_bias = (
            np.zeros(7, dtype=np.float64)
            if self.control_mode == "position"
            else self.estimate_gravity_torque(initial_q)
        )
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
                # Disable just the iiwa customization script, not all model
                # scripts.  The script enabled parameter is the supported
                # per-script switch in current CoppeliaSim releases.
                enabled_param = getattr(self.sim, "scriptintparam_enabled", None)
                if enabled_param is not None:
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
        self.active = True

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
        # When the arm is nearly stationary, the PD effort needed to hold the
        # target is a useful local estimate of gravity plus static friction.
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
        limits = np.asarray(self.config.torque_limits, dtype=np.float64)
        for joint, position, force_limit in zip(self.joints, self.qd, limits):
            self.sim.setJointTargetPosition(joint, float(position))
            try:
                self.sim.setJointTargetForce(joint, float(abs(force_limit)), True)
            except TypeError:
                self.sim.setJointTargetForce(joint, float(abs(force_limit)))
        self.last_torque = np.zeros(7, dtype=np.float64)
        return self.last_torque.copy(), q, dq

    def step(self, qd: Sequence[float] | None = None, steps: int | None = None) -> dict[str, Any]:
        if qd is not None:
            self.set_target(qd)
        count = max(1, int(steps if steps is not None else self.config.substeps_per_observation))
        for _ in range(count):
            if self.control_mode == "position":
                torque, q, dq = self.apply_position_target()
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
        if not self.active:
            self.configure_dynamic_mode()
        self.client.setStepping(True)
        state = self.sim.getSimulationState()
        if state in {self.sim.simulation_stopped, self.sim.simulation_paused}:
            self.sim.startSimulation()
            self.started_simulation = True

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
            "dynamic_dt_s": self.dynamic_dt,
            "joint_modes": [_joint_mode(self.sim, joint) for joint in self.joints],
            "dynctrl_modes": [self._read_dynctrl_mode(joint) for joint in self.joints],
            "q": self.read_positions().tolist(),
            "dq": self.read_velocities().tolist(),
            "last_torque": self.last_torque.tolist(),
            "gravity_bias": self.bias.tolist(),
            "gravity_bias_estimate": self.gravity_bias.tolist(),
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
        if self.iiwa_script is not None and self._original_scripts_inactive:
            try:
                enabled_param = getattr(self.sim, "scriptintparam_enabled", None)
                if enabled_param is not None:
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
            self.restore_joint_modes()
