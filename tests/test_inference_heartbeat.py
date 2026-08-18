import asyncio
import threading
import time

import pytest
from openhands.sdk import LLM
from pydantic import SecretStr

from senpai_agent.inference_heartbeat import InferenceTrackedLLM


def wait_until(predicate, timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError("inference heartbeat condition was not met")


def tracked_llm(callback) -> InferenceTrackedLLM:
    llm = InferenceTrackedLLM(
        model="anthropic/claude-opus-4-8",
        api_key=SecretStr("test-key"),
    )
    llm.configure_inference_heartbeat(callback, interval_seconds=0.01)
    return llm


def test_model_copies_share_a_live_inference_heartbeat(monkeypatch):
    entered = {name: threading.Event() for name in ("first", "second")}
    release = {name: threading.Event() for name in ("first", "second")}
    states = []
    state_lock = threading.Lock()

    def callback(started_at, heartbeat_at):
        with state_lock:
            states.append((started_at, heartbeat_at))

    def completion(_llm, messages, *_args, **_kwargs):
        name = messages[0]
        entered[name].set()
        assert release[name].wait(2)
        return name

    monkeypatch.setattr(LLM, "completion", completion)
    llm = tracked_llm(callback)
    copied_llm = llm.model_copy()
    results = []
    first = threading.Thread(
        target=lambda: results.append(llm.completion(["first"]))
    )
    second = threading.Thread(
        target=lambda: results.append(copied_llm.completion(["second"]))
    )

    first.start()
    second.start()
    assert entered["first"].wait(2)
    assert entered["second"].wait(2)
    with state_lock:
        state_count = len(states)
        previous_heartbeat = states[-1][1]
    wait_until(
        lambda: len(states) > state_count
        and states[-1][1] is not None
        and states[-1][1] > previous_heartbeat
    )
    with state_lock:
        active_states = [state for state in states if state[0] is not None]
    assert len({state[0] for state in active_states}) == 1

    release["first"].set()
    first.join(2)
    with state_lock:
        assert states[-1][0] is not None

    release["second"].set()
    second.join(2)
    with state_lock:
        assert states[-1] == (None, None)
    assert sorted(results) == ["first", "second"]


def test_async_inference_failure_clears_the_heartbeat(monkeypatch):
    states = []

    async def acompletion(_llm, *_args, **_kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(LLM, "acompletion", acompletion)
    llm = tracked_llm(lambda *state: states.append(state))

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(llm.acompletion([]))

    assert states[0][0] is not None
    assert states[-1] == (None, None)
