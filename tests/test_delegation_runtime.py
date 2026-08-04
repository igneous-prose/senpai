import threading
from types import SimpleNamespace

from openhands.sdk import LLM, Agent
from pydantic import SecretStr

from senpai_agent.openhands_runner import with_tool_concurrency


def test_file_agent_runtime_executor_honors_the_visible_parallel_limit():
    agent = Agent(
        llm=LLM(
            model="openai/gpt-5.6-sol",
            api_key=SecretStr("test-key"),
        ),
        tools=[],
        tool_concurrency_limit=1,
    )
    agent = with_tool_concurrency(agent, 2)
    both_started = threading.Event()
    lock = threading.Lock()
    started = 0

    def run(_action):
        nonlocal started
        with lock:
            started += 1
            if started == 2:
                both_started.set()
        assert both_started.wait(1), "tool calls did not overlap"
        return []

    results = agent._parallel_executor.execute_batch(
        [SimpleNamespace(tool_name="first"), SimpleNamespace(tool_name="second")],
        run,
    )

    assert agent.tool_concurrency_limit == 2
    assert results == [[], []]
