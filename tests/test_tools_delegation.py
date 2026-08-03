import threading
import uuid
from types import SimpleNamespace

import pytest
from openhands.sdk.context.view import View
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent

from senpai_agent.delegation import (
    DelegateAgentAction,
    DelegateAgentTool,
    DelegationRequest,
)


class EventSink:
    def __init__(self):
        self.events = []
        self.received = threading.Event()

    def enqueue(self, event) -> bool:
        self.events.append(event)
        self.received.set()
        return True


class FakeChild:
    def __init__(
        self,
        *,
        release: threading.Event,
        fail: bool = False,
    ):
        self.release = release
        self.fail = fail
        self.started = threading.Event()
        self.calls: list[tuple[str, float | None]] = []
        self.interrupted = False

    def run(self, task: str, timeout_seconds: float | None) -> str:
        self.calls.append((task, timeout_seconds))
        self.started.set()
        self.release.wait()
        if self.fail:
            raise RuntimeError("child disappeared")
        return "Child research result"

    def interrupt(self) -> None:
        self.interrupted = True
        self.release.set()


def parent_conversation() -> SimpleNamespace:
    view = View(
        events=[
            MessageEvent(
                source="user",
                llm_message=Message(
                    role="user",
                    content=[TextContent(text="Investigate the regression")],
                ),
                activated_skills=["experiment-report"],
                extended_content=[
                    TextContent(text="Progressively disclosed skill instructions")
                ],
            ),
            MessageEvent(
                source="agent",
                llm_message=Message(
                    role="assistant",
                    content=[TextContent(text="I will inspect the evidence.")],
                ),
            ),
        ]
    )
    return SimpleNamespace(id=uuid.uuid4(), state=SimpleNamespace(view=view))


def make_delegate(
    factory,
    sink: EventSink,
    *,
    max_workers: int = 8,
    max_runtime_seconds: float | None = None,
    background_allowed: bool = True,
):
    return DelegateAgentTool.create(
        child_runner_factory=factory,
        event_sink=sink,
        max_workers=max_workers,
        max_runtime_seconds=max_runtime_seconds,
        background_allowed=background_allowed,
    )[0]


def test_foreground_delegation_returns_the_result_and_runtime_inline():
    release = threading.Event()
    release.set()
    child = FakeChild(release=release)
    delegate = make_delegate(
        lambda _request: child,
        EventSink(),
        max_runtime_seconds=12,
    )

    try:
        observation = delegate(
            DelegateAgentAction(
                task="Locate the implementation.",
                agent="explore",
                model="fast",
            ),
            parent_conversation(),
        )

        assert observation.status == "finished"
        assert observation.result == "Child research result"
        assert child.calls == [("Locate the implementation.", 12)]
    finally:
        delegate.executor.close()


def test_frontier_delegation_defaults_to_the_general_purpose_agent():
    release = threading.Event()
    release.set()
    child = FakeChild(release=release)
    requests: list[DelegationRequest] = []

    def factory(request: DelegationRequest) -> FakeChild:
        requests.append(request)
        return child

    delegate = make_delegate(factory, EventSink())

    try:
        observation = delegate(
            DelegateAgentAction(
                task="Reconsider the research direction and implement the best fix.",
                model="frontier",
            ),
            parent_conversation(),
        )

        assert observation.status == "finished"
        assert requests[0].model == "frontier"
        assert requests[0].agent == "general-purpose"
    finally:
        delegate.executor.close()


@pytest.mark.parametrize("include_context", [False, True])
def test_background_delegation_copies_parent_context_only_when_requested(
    include_context: bool,
):
    parent = parent_conversation()
    release = threading.Event()
    child = FakeChild(release=release)
    requests: list[DelegationRequest] = []
    sink = EventSink()

    def factory(request: DelegationRequest) -> FakeChild:
        requests.append(request)
        return child

    delegate = make_delegate(factory, sink)

    try:
        observation = delegate(
            DelegateAgentAction(
                task="Compare the candidate runs.",
                agent="explore",
                model="fast",
                background=True,
                include_context=include_context,
            ),
            parent,
        )

        assert observation.status == "dispatched"
        assert child.started.wait(1)
        assert not sink.received.is_set()
        request = requests[0]
        assert request.task_id == observation.task_id
        assert request.parent_conversation_id == str(parent.id)
        if include_context:
            assert [message.role for message in request.parent_context] == [
                "user",
                "assistant",
            ]
            assert [
                content.text
                for content in request.parent_context[0].content
                if isinstance(content, TextContent)
            ] == [
                "Investigate the regression",
                "Progressively disclosed skill instructions",
            ]
        else:
            assert request.parent_context == ()

        release.set()
        assert sink.received.wait(1)
        event = sink.events[0]
        assert event.kind == "agent_result"
        assert event.dedupe_key == f"agent_result:{observation.task_id}"
        assert event.payload == {
            "task_id": observation.task_id,
            "parent_conversation_id": str(parent.id),
            "task": "Compare the candidate runs.",
            "result": "Child research result",
        }
    finally:
        release.set()
        delegate.executor.close()


def test_background_delegation_reports_child_failures_as_durable_events():
    release = threading.Event()
    release.set()
    child = FakeChild(release=release, fail=True)
    sink = EventSink()
    delegate = make_delegate(lambda _request: child, sink)
    parent = parent_conversation()

    try:
        observation = delegate(
            DelegateAgentAction(
                task="Check one hypothesis.",
                background=True,
            ),
            parent,
        )

        assert sink.received.wait(1)
        event = sink.events[0]
        assert event.kind == "agent_error"
        assert event.dedupe_key == f"agent_result:{observation.task_id}"
        assert event.payload == {
            "task_id": observation.task_id,
            "parent_conversation_id": str(parent.id),
            "task": "Check one hypothesis.",
            "error": "RuntimeError: child disappeared",
        }
    finally:
        delegate.executor.close()


def test_background_worker_slot_is_reusable_after_interrupt():
    releases = [threading.Event(), threading.Event()]
    children = []
    sink = EventSink()

    def factory(_request: DelegationRequest) -> FakeChild:
        child = FakeChild(release=releases[len(children)])
        children.append(child)
        return child

    delegate = make_delegate(factory, sink, max_workers=1)
    parent = parent_conversation()

    try:
        delegate(
            DelegateAgentAction(task="First task.", background=True),
            parent,
        )
        assert children[0].started.wait(1)
        with pytest.raises(
            RuntimeError,
            match=r"subagent capacity is full \(1 active\)",
        ):
            delegate(
                DelegateAgentAction(task="Blocked task.", background=True),
                parent,
            )

        delegate.executor.interrupt()
        assert sink.received.wait(1)
        assert children[0].interrupted is True

        sink.received.clear()
        releases[1].set()
        observation = delegate(
            DelegateAgentAction(task="Next task.", background=True),
            parent,
        )
        assert sink.received.wait(1)
        assert observation.status == "dispatched"
        assert len(sink.events) == 2
    finally:
        for release in releases:
            release.set()
        delegate.executor.close()


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (
            DelegateAgentAction(task="Search.", agent="search"),
            "search_mode is required",
        ),
        (
            DelegateAgentAction(
                task="Explore.",
                agent="explore",
                search_mode="general-web",
            ),
            "search_mode is required only",
        ),
        (
            DelegateAgentAction(task="Nested.", background=True),
            "background=false",
        ),
    ],
)
def test_invalid_delegations_fail_before_starting_a_child(action, message: str):
    starts = []
    delegate = make_delegate(
        lambda request: starts.append(request),
        EventSink(),
        background_allowed=False,
    )

    try:
        with pytest.raises(ValueError, match=message):
            delegate(action, parent_conversation())
        assert starts == []
    finally:
        delegate.executor.close()


def test_delegation_requires_its_parent_conversation():
    starts = []
    delegate = make_delegate(
        lambda request: starts.append(request),
        EventSink(),
    )

    try:
        with pytest.raises(ValueError, match="parent conversation"):
            delegate(DelegateAgentAction(task="Orphan task."))
        assert starts == []
    finally:
        delegate.executor.close()
