"""Managed CoppeliaSim ZeroMQ sessions.

The stock ``RemoteAPIClient`` closes its socket in ``__del__``.  That is
usually fine for ordinary RPC calls, but it can leave the CoppeliaSim ZMQ
server coroutine suspended when a process exits after using synchronous
stepping.  CoppeliaSim then reports the otherwise confusing
``ZMQ remote API server@addOnScript: abort execution`` notification.

This wrapper sends the server's explicit ``_*end*_`` request before closing
the socket.  It is intentionally small and keeps the public API identical to
the bundled client, so existing perception and control code does not need to
know about session cleanup.
"""

from __future__ import annotations

import atexit
import threading
import weakref
from typing import Any

from coppeliasim_zmqremoteapi_client import RemoteAPIClient as _RemoteAPIClient


_clients: "weakref.WeakSet[ManagedRemoteAPIClient]" = weakref.WeakSet()
_clients_lock = threading.Lock()


class ManagedRemoteAPIClient(_RemoteAPIClient):
    """RemoteAPIClient with an idempotent, graceful disconnect."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._managed_closed = False
        self._managed_stepping = False
        with _clients_lock:
            _clients.add(self)

    def setStepping(self, enable: bool = True):  # noqa: N802 - API compatibility
        result = super().setStepping(bool(enable))
        self._managed_stepping = bool(enable)
        return result

    def close(self) -> None:
        """Release stepping and tell the server this client is leaving."""

        if getattr(self, "_managed_closed", False):
            return
        self._managed_closed = True
        try:
            # A pending synchronous step must be released before the client
            # socket is destroyed.  This call is harmless when stepping was
            # not enabled and is safe to repeat on a stopped simulation.
            if getattr(self, "sendCnt", 0) > 0 and self._managed_stepping:
                try:
                    super().setStepping(False)
                except Exception:
                    pass
                self._managed_stepping = False

            # ``_*end*_`` is handled specially by the CoppeliaSim add-on: it
            # removes this UUID from its client table and returns an empty
            # response (not a normal ``ret`` response), so use the low-level
            # send/receive pair rather than ``call``.
            if getattr(self, "sendCnt", 0) > 0:
                try:
                    self._send({"func": "_*end*_", "args": []})
                    self._recv()
                except Exception:
                    pass
        finally:
            try:
                self.socket.close(0)
            except Exception:
                try:
                    self.socket.close()
                except Exception:
                    pass
            try:
                self.context.term()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# Keep the existing import spelling in all project scripts.
RemoteAPIClient = ManagedRemoteAPIClient


def _close_all() -> None:
    with _clients_lock:
        clients = list(_clients)
    for client in clients:
        try:
            client.close()
        except Exception:
            pass


atexit.register(_close_all)

