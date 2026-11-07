"""Client for environment rollout operations over WebSocket.

Connection direction: THIS process LISTENS and the rollout client
(`client.run_client`) dials in. We listen because the robot workstation can't
accept inbound connections while the learner/eval host can — so the data path is
a direct, app-owned TCP socket where TCP_NODELAY is effective and per-step
latency stays flat. The message protocol is unchanged: this side SENDS operation
requests (create_env/reset/step/...) and the rollout side replies.
"""

import logging
import socket
import threading
import time
from typing import Dict, Tuple, Any

import numpy as np
import websockets.sync.server
import websockets.exceptions
from openpi_client import msgpack_numpy

# Errors during an operation that just mean the connection dropped: close it and
# wait for the rollout client to redial, then retry.
_RETRY_EXC: tuple = (OSError, websockets.exceptions.ConnectionClosed)


class EnvClient:
    """Environment operations over a single persistent WebSocket.

    Listens on host:port; the rollout client dials in. Re-accepts automatically
    if the connection drops.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8102):
        self.host = host
        self.port = port
        self._packer = msgpack_numpy.Packer()
        # Termination/detection piggybacked on the last get_observation() response,
        # consumed by get_info_for_step() so it needs no separate round-trip.
        self._last_info = None
        # The accept server runs in a background thread and hands each incoming
        # connection to the calling loop via this condition/slot.
        self._server = None
        self._conn = None
        # Accumulated time (ms) spent blocked in conn.recv() waiting for the rollout
        # client's reply; the training loop pops this once per step for timing logs.
        self.network_ms = 0.0
        self._cond = threading.Condition()
        self._pending = None       # (conn, done_event) accepted, awaiting adoption
        self._active_done = None   # done_event that keeps the current handler alive

    @staticmethod
    def _set_nodelay(conn):
        """Disable Nagle: per-step RPCs are tiny frames that would otherwise stall
        ~40ms on the TCP delayed-ACK timer."""
        try:
            conn.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as e:
            logging.warning("Could not set TCP_NODELAY: %s", e)

    def _ensure_server(self):
        if self._server is not None:
            return

        def _handler(conn):
            self._set_nodelay(conn)
            done = threading.Event()
            with self._cond:
                self._pending = (conn, done)
                self._cond.notify_all()
            done.wait()  # hold the connection open until the loop releases it

        self._server = websockets.sync.server.serve(
            _handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            close_timeout=100,
        )
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        logging.info("EnvClient listening for rollout client on %s:%s", self.host, self.port)

    def _accept(self):
        """Block until the rollout client dials in; return its connection."""
        self._ensure_server()
        with self._cond:
            while self._pending is None:
                self._cond.wait()
            conn, done = self._pending
            self._pending = None
        self._active_done = done
        return conn

    def _get_connection(self):
        """Return the current connection, accepting one if needed (persistent)."""
        if self._conn is None:
            self._conn = self._accept()
        return self._conn

    def _close_connection(self):
        """Drop the current connection so the next call re-accepts."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        # Release the handler thread so the connection is torn down server-side and
        # a fresh dial-in can be accepted.
        if self._active_done is not None:
            self._active_done.set()
            self._active_done = None

    def close(self):
        """Drop the connection and stop the accept server (used on env recovery)."""
        self._close_connection()
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None

    def _call_operation(self, operation: str, request: dict) -> dict:
        last_exc = None
        for attempt in range(3):
            try:
                conn = self._get_connection()
                conn.send(self._packer.pack({"operation": operation, **request}))
                recv_start = time.time()
                raw = conn.recv()
                self.network_ms += (time.time() - recv_start) * 1000.0
                response = msgpack_numpy.unpackb(raw)
                if response.get("status", response.get("stats")) == "error":
                    raise RuntimeError(
                        f"Environment operation {operation} failed: {response.get('message')}"
                    )
                return response
            except _RETRY_EXC as e:
                last_exc = e
                self._close_connection()
                if attempt < 2:
                    logging.debug(
                        "EnvClient %s connection dropped (attempt %s), re-accepting...",
                        operation, attempt + 1,
                    )
                    time.sleep(2)
            except Exception as e:
                self._close_connection()
                raise RuntimeError(f"EnvClient {operation} failed: {e}") from e
        raise RuntimeError(f"EnvClient {operation} failed after retries: {last_exc}") from last_exc

    def _prepare_request(self, request: dict) -> dict:
        """Convert numpy arrays to lists for serialization."""
        prepared = {}
        for k, v in request.items():
            prepared[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return prepared

    def create_env(self, request: dict) -> Tuple[str, str]:
        """Create an environment."""
        response = self._call_operation("create_env", self._prepare_request(request))
        return response["env_id"], response["task_description"]

    def reset(self, env_id: str) -> Tuple[Dict[str, Any], bool]:
        """Reset an environment."""
        response = self._call_operation("reset", {"env_id": env_id})
        return response["observation"], response["done"]

    def step(self, env_id: str, action: np.ndarray) -> Tuple[np.ndarray, str]:
        """Step the environment. Returns (real_executed_action, action_type)."""
        response = self._call_operation("step", {"env_id": env_id, "action": action})
        real_action = np.array(response.get("action", action))
        action_type = response.get("action_type", "policy")
        return real_action, action_type

    def get_observation(self, env_id: str) -> dict:
        """Get the observation of the environment."""
        response = self._call_operation("get_observation", {"env_id": env_id})
        # Termination/detection is piggybacked on the obs response (server computes it on
        # the same frame). Cache it so get_info_for_step needs no separate round-trip.
        if "done" in response:
            self._last_info = (
                response["done"], response["success"],
                response["reward"], response["mask"],
            )
        return response["observation"]

    def get_info_for_step(self, env_id: str) -> Tuple[bool, bool, float, float]:
        """Evaluate termination after a step: (done, success, reward, continuation_mask).

        Returns the values piggybacked on the most recent get_observation() (no network
        round-trip). Falls back to an RPC only if no observation was fetched first."""
        if self._last_info is not None:
            info = self._last_info
            self._last_info = None
            return info
        response = self._call_operation("get_info_for_step", {"env_id": env_id})
        return response["done"], response["success"], response["reward"], response["mask"]

    def render(self, env_id: str) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB frame of the current env state (for eval videos)."""
        response = self._call_operation("render", {"env_id": env_id})
        return np.asarray(response["frame"])


class EnvClientWrapper:
    """Gym-like interface around EnvClient."""

    def __init__(self, env_creation_request: dict, host: str = "0.0.0.0", port: int = 8102):
        """Initialize the wrapper. Binds host:port and waits for the rollout client
        to dial in (see EnvClient)."""
        self.host = host
        self.port = port
        self.client = EnvClient(host=host, port=port)
        self.env_id, self.task_description = self.client.create_env(env_creation_request)
        self.env_creation_request = env_creation_request

    def _call(self, op_name: str, thunk):
        """Run an op; if the server returns an error, recreate env + reset, then retry."""
        status = "normal"
        while True:
            if status == "normal":
                try:
                    return thunk()
                except RuntimeError as e:
                    logging.warning(f"{op_name} failed ({e}); will recreate env and retry...")
                    status = "recover"
                    time.sleep(10)
            else:
                try:
                    try:
                        self.client.close()
                    except Exception:
                        pass
                    self.client = EnvClient(host=self.host, port=self.port)
                    self.env_id, self.task_description = self.client.create_env(self.env_creation_request)
                    if op_name != "reset":
                        self.client.reset(self.env_id)
                    self.client.reset(self.env_id)
                    time.sleep(60)
                    status = "normal"
                except RuntimeError as e:
                    logging.warning(f"{op_name} recovery failed ({e}); retrying recovery...")
                    time.sleep(10)

    def reset(self):
        """Reset the environment and return observation."""
        observation, _ = self._call("reset", lambda: self.client.reset(self.env_id))
        return observation

    def step(self, action):
        """Step the environment. Returns (real_executed_action, action_type)."""
        return self._call("step", lambda: self.client.step(self.env_id, action))

    def get_observation(self):
        """Get the observation of the environment."""
        return self._call("get_observation", lambda: self.client.get_observation(self.env_id))

    def get_info_for_step(self):
        """Evaluate termination after a step: (done, success, reward, continuation_mask)."""
        return self._call("get_info_for_step", lambda: self.client.get_info_for_step(self.env_id))

    def pop_network_ms(self):
        """Return time (ms) spent waiting for client replies since the last call, and reset."""
        ms = self.client.network_ms
        self.client.network_ms = 0.0
        return ms

    def render(self):
        """Return an (H, W, 3) uint8 RGB frame. Deliberately NOT wrapped in `_call`'s
        env-recreation recovery — a render failure (e.g. an old server without the op) should
        propagate so the caller can disable video, not trigger an env rebuild loop."""
        return self.client.render(self.env_id)
