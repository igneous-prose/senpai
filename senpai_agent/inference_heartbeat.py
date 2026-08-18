"""Publish model-request activity without adding conversation events."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from openhands.sdk import LLM
from pydantic import PrivateAttr


InferenceHeartbeatCallback = Callable[[float | None, float | None], None]


class InferenceHeartbeat:
    """Track active requests and publish their earliest start plus a live pulse."""

    def __init__(
        self,
        callback: InferenceHeartbeatCallback,
        *,
        interval_seconds: float = 30,
    ):
        if interval_seconds <= 0:
            raise ValueError("inference heartbeat interval must be positive")
        self.callback = callback
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._active: dict[object, float] = {}

    @contextmanager
    def request(self) -> Iterator[None]:
        token = object()
        stop = threading.Event()
        with self._lock:
            self._active[token] = time.time()
            self._publish_locked()
        threading.Thread(
            target=self._pulse,
            args=(token, stop),
            name="senpai-inference-heartbeat",
            daemon=True,
        ).start()
        try:
            yield
        finally:
            stop.set()
            with self._lock:
                self._active.pop(token, None)
                self._publish_locked()

    def _pulse(self, token: object, stop: threading.Event) -> None:
        while not stop.wait(self.interval_seconds):
            with self._lock:
                if token not in self._active:
                    return
                self._publish_locked()

    def _publish_locked(self) -> None:
        if not self._active:
            self.callback(None, None)
            return
        self.callback(min(self._active.values()), time.time())


class InferenceTrackedLLM(LLM):
    """Wrap every provider request in an out-of-band inference heartbeat."""

    _inference_heartbeat: InferenceHeartbeat | None = PrivateAttr(default=None)

    def configure_inference_heartbeat(
        self,
        callback: InferenceHeartbeatCallback,
        *,
        interval_seconds: float = 30,
    ) -> None:
        self._inference_heartbeat = InferenceHeartbeat(
            callback,
            interval_seconds=interval_seconds,
        )

    @contextmanager
    def _inference_request(self) -> Iterator[None]:
        if self._inference_heartbeat is None:
            yield
            return
        with self._inference_heartbeat.request():
            yield

    def completion(self, *args: Any, **kwargs: Any) -> Any:
        with self._inference_request():
            return super().completion(*args, **kwargs)

    async def acompletion(self, *args: Any, **kwargs: Any) -> Any:
        with self._inference_request():
            return await super().acompletion(*args, **kwargs)

    def responses(self, *args: Any, **kwargs: Any) -> Any:
        with self._inference_request():
            return super().responses(*args, **kwargs)

    async def aresponses(self, *args: Any, **kwargs: Any) -> Any:
        with self._inference_request():
            return await super().aresponses(*args, **kwargs)
