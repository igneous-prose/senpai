import json
import subprocess
from pathlib import Path
from uuid import UUID

from pydantic import SecretStr

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.controller import (
    Controller,
    TurnResult,
    _full_prompt,
)
from senpai_agent.github_mailbox import GitHubMailbox
from senpai_agent.mailbox import (
    CompositeMailbox,
    ControllerEvent,
    LocalStudentMailbox,
)
from senpai_agent.models import AssignmentRecord, render_assignment_marker
from senpai_agent.monitor import (
    MonitorMailbox,
    MonitorEvaluation,
    MonitorSignal,
    MonitorStore,
    TrainingMonitorSpec,
)
from senpai_agent.state import (
    AssignmentConversationRegistry,
    ConversationStateLedger,
    StudentConversationSelector,
)
from senpai_agent.supervisor import ProgressLease, WorkerLease
from senpai_agent.training import TrainingState
from senpai_agent.workspace import StudentWorkspaceReconciler


class Mailbox:
    def __init__(self, polls):
        self.polls = list(polls)
        self.calls = 0
        self.acknowledged = []

    def poll(self):
        self.calls += 1
        return self.polls.pop(0) if self.polls else ()

    def acknowledge(self, dedupe_keys):
        self.acknowledged.append(tuple(dedupe_keys))


class Turns:
    def __init__(self):
        self.calls = []

    def run(
        self,
        prompt,
        *,
        conversation_id,
        event_keys,
    ):
        self.calls.append((prompt, conversation_id, event_keys))
        return TurnResult(exit_code=0)


class SequencedTurns(Turns):
    def __init__(self, exit_codes):
        super().__init__()
        self.exit_codes = iter(exit_codes)

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        return TurnResult(
            exit_code=next(self.exit_codes),
            delivered_event_keys=result.delivered_event_keys,
        )


class RaisingTurns(Turns):
    def __init__(self):
        super().__init__()
        self.fail = True

    def run(self, *args, **kwargs):
        if self.fail:
            self.fail = False
            raise RuntimeError("SDK transport failed")
        return super().run(*args, **kwargs)


def test_assignment_conversation_is_reused_for_monitor_wake(tmp_path: Path):
    registry = AssignmentConversationRegistry(tmp_path / "students.json")

    first = registry.for_assignment("assignment-1", "revision-2")
    wake = registry.for_assignment("assignment-1", "revision-2")
    next_revision = registry.for_assignment("assignment-1", "revision-3")

    assert isinstance(first, UUID)
    assert wake == first
    assert next_revision != first


def test_late_student_child_result_wakes_its_exact_parent(tmp_path: Path):
    first_parent = UUID("00000000-0000-0000-0000-000000000011")
    second_parent = UUID("00000000-0000-0000-0000-000000000012")
    store_path = tmp_path / "student-events.sqlite3"
    with AdvisorEventStore(store_path) as store:
        store.enqueue(
            AdvisorEvent(
                kind="agent_result",
                dedupe_key="agent_result:first",
                payload={"parent_conversation_id": str(first_parent)},
            )
        )
        store.enqueue(
            AdvisorEvent(
                kind="agent_result",
                dedupe_key="agent_result:second",
                payload={"parent_conversation_id": str(second_parent)},
            )
        )

    events = LocalStudentMailbox(store_path).poll()
    batches = StudentConversationSelector(
        AssignmentConversationRegistry(tmp_path / "students.json")
    )(events)

    assert len(events) == 1
    assert events[0].payload["count"] == 1
    assert events[0].payload["conversation_id"] == str(first_parent)
    assert len(batches) == 1
    assert batches[0].conversation_id == first_parent


def test_controller_builds_first_turn_from_program_and_role_task(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    instructions = workspace / "instructions"
    instructions.mkdir(parents=True)
    (workspace / "program.md").write_text("Minimize test error.")
    (instructions / "prompt-student.md").write_text(
        "Work as $STUDENT_NAME on $ADVISOR_BRANCH."
    )
    env = {
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "GH_REPO": "acme/widgets",
        "ADVISOR_BRANCH": "research",
        "WANDB_ENTITY": "acme",
        "WANDB_PROJECT": "cfd",
        "STUDENT_NAME": "fern",
    }

    prompt = _full_prompt("student", env)

    assert "Minimize test error." in prompt
    assert "Work as fern on research." in prompt
    assert "Role: student; repository: acme/widgets" in prompt


def test_advisor_repolls_immediately_after_a_turn_when_github_changed():
    review = ControllerEvent(
        kind="review_ready",
        dedupe_key="review_ready:17:abc",
        payload={"pr": 17, "head_sha": "abc"},
    )
    mailbox = Mailbox(
        [
            (
                ControllerEvent(
                    kind="idle_student",
                    dedupe_key="idle:student-1",
                    payload={"student": "student-1"},
                ),
            ),
            (review,),
            (),
        ]
    )
    turns = Turns()
    controller = Controller(
        role="advisor",
        mailbox=mailbox,
        turns=turns,
        conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
        full_prompt="programme",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=2)

    assert len(turns.calls) == 2
    assert "idle_student" in turns.calls[0][0]
    assert "Current time (UTC):" in turns.calls[0][0]
    assert "review_ready" in turns.calls[1][0]
    assert mailbox.calls >= 3
    assert "programme" not in turns.calls[1][0]


def test_no_github_work_means_no_model_turn():
    turns = Turns()
    controller = Controller(
        role="student",
        mailbox=Mailbox([(), ()]),
        turns=turns,
        conversation_id=UUID("00000000-0000-0000-0000-000000000002"),
        full_prompt="programme",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=2)

    assert turns.calls == []


def test_controller_retries_an_unacknowledged_event_after_turn_failure():
    event = ControllerEvent(
        kind="review_ready",
        dedupe_key="review:17:abc",
        payload={"number": 17},
    )
    turns = SequencedTurns([1, 0])
    controller = Controller(
        role="advisor",
        mailbox=Mailbox([(event,), (event,), ()]),
        turns=turns,
        conversation_id=UUID("00000000-0000-0000-0000-000000000003"),
        full_prompt="programme",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=2)

    assert len(turns.calls) == 2
    assert "programme" in turns.calls[1][0]
    assert all("programme" in call[0] for call in turns.calls)


def test_controller_retries_an_unacknowledged_event_after_sdk_exception():
    event = ControllerEvent(
        kind="review_ready",
        dedupe_key="review:17:abc",
        payload={"number": 17},
    )
    turns = RaisingTurns()
    controller = Controller(
        role="advisor",
        mailbox=Mailbox([(event,), (event,), ()]),
        turns=turns,
        conversation_id=UUID("00000000-0000-0000-0000-000000000005"),
        full_prompt="programme",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=2)

    assert len(turns.calls) == 1
    assert "programme" in turns.calls[0][0]


def test_student_events_are_delivered_and_acknowledged_per_conversation(
    tmp_path: Path,
):
    first = UUID("00000000-0000-0000-0000-000000000081")
    second = UUID("00000000-0000-0000-0000-000000000082")
    first_event = ControllerEvent(
        kind="training_monitor",
        dedupe_key="monitor:first",
        payload={"conversation_id": str(first), "summary": "first only"},
    )
    second_event = ControllerEvent(
        kind="local_events_pending",
        dedupe_key="child:second",
        payload={"conversation_id": str(second), "summary": "second only"},
    )
    mailbox = Mailbox([(first_event, second_event), ()])
    turns = Turns()
    controller = Controller(
        role="student",
        mailbox=mailbox,
        turns=turns,
        conversation_id=first,
        full_prompt="programme",
        conversation_for_events=StudentConversationSelector(
            AssignmentConversationRegistry(tmp_path / "students.json")
        ),
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    assert [call[1] for call in turns.calls] == [first, second]
    assert "first only" in turns.calls[0][0]
    assert "second only" not in turns.calls[0][0]
    assert "second only" in turns.calls[1][0]
    assert "first only" not in turns.calls[1][0]
    assert mailbox.acknowledged == [("monitor:first",), ("child:second",)]


def test_one_student_conversation_failure_does_not_ack_or_starve_another(
    tmp_path: Path,
):
    first = UUID("00000000-0000-0000-0000-000000000083")
    second = UUID("00000000-0000-0000-0000-000000000084")
    first_event = ControllerEvent(
        kind="training_monitor",
        dedupe_key="monitor:first",
        payload={"conversation_id": str(first)},
    )
    second_event = ControllerEvent(
        kind="training_monitor",
        dedupe_key="monitor:second",
        payload={"conversation_id": str(second)},
    )
    mailbox = Mailbox([(first_event, second_event)])
    turns = SequencedTurns([1, 0])
    controller = Controller(
        role="student",
        mailbox=mailbox,
        turns=turns,
        conversation_id=first,
        full_prompt="programme",
        conversation_for_events=StudentConversationSelector(
            AssignmentConversationRegistry(tmp_path / "students.json")
        ),
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    assert [call[1] for call in turns.calls] == [first, second]
    assert mailbox.acknowledged == [("monitor:second",)]


def test_composite_mailbox_keeps_healthy_events_when_a_peer_fails(capsys):
    event = ControllerEvent(
        kind="student_assignment",
        dedupe_key="assignment:healthy",
        payload={"assignment_id": "healthy"},
    )

    class BrokenMailbox:
        def poll(self):
            raise RuntimeError("monitor backend unavailable")

        def acknowledge(self, _dedupe_keys):
            return

    mailbox = CompositeMailbox(BrokenMailbox(), Mailbox([(event,)]))

    assert mailbox.poll() == (event,)
    assert "SENPAI_MAILBOX_ERROR RuntimeError" in capsys.readouterr().err


def test_every_monitor_signal_directly_wakes_its_student(tmp_path: Path):
    conversation_id = UUID("00000000-0000-0000-0000-000000000086")
    signal = MonitorSignal(
        kind="training_status",
        dedupe_key="training:failed",
        training_id="training-1",
        state=TrainingState.FAILED,
        detail="training failed",
    )
    store = MonitorStore(tmp_path / "monitors.sqlite3")
    spec = TrainingMonitorSpec(
        training_id="training-1",
        conversation_id=conversation_id,
    )
    store.register(spec)
    store.record_poll(spec, MonitorEvaluation(signals=(signal,)), None)

    class Engine:
        def poll(self):
            return ()

    events = MonitorMailbox(Engine(), store).poll()

    assert len(events) == 1
    assert events[0].payload["conversation_id"] == str(conversation_id)
    assert events[0].payload["summary"] == "training failed"
    assert "registered monitor policy" in str(events[0].payload["reason"])
    store.close()


def test_controller_waits_behind_start_gate_while_refreshing_its_lease(
    tmp_path: Path,
):
    gate = tmp_path / "start-gate"
    lease_path = tmp_path / "controller-lease.json"
    mailbox = Mailbox([()])
    sleeps = []

    def open_gate(seconds):
        sleeps.append(seconds)
        gate.write_text("open")

    controller = Controller(
        role="advisor",
        mailbox=mailbox,
        turns=Turns(),
        conversation_id=UUID("00000000-0000-0000-0000-000000000085"),
        full_prompt="programme",
        progress=ProgressLease(lease_path),
        start_gate_path=gate,
        start_gate_poll_seconds=7,
        sleep=open_gate,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    assert sleeps == [7]
    assert mailbox.calls == 1
    lease = WorkerLease.read(lease_path)
    assert lease.phase == "poll"


def test_assignment_reconciliation_preserves_unpushed_commits_on_restart(
    tmp_path: Path,
):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    workspace = tmp_path / "student"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    subprocess.run(["git", "init", str(seed)], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(seed), "config", "user.email", "test@example.com"],
        check=True,
    )
    (seed / "program.py").write_text("baseline\n")
    subprocess.run(["git", "-C", str(seed), "add", "program.py"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-m", "baseline"], check=True)
    subprocess.run(
        ["git", "-C", str(seed), "branch", "-M", "student/candidate"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(seed), "remote", "add", "origin", str(remote)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(seed), "push", "origin", "student/candidate"],
        check=True,
    )
    remote_head = subprocess.run(
        ["git", "-C", str(seed), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "clone", "--branch", "student/candidate", str(remote), str(workspace)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.name", "student"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.email", "student@example.com"],
        check=True,
    )
    (workspace / "program.py").write_text("candidate\n")
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "-am", "candidate"], check=True
    )
    local_head = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    (workspace / "notes.txt").write_text("dirty but recoverable\n")
    event = ControllerEvent(
        kind="student_assignment",
        dedupe_key="assignment:restart",
        payload={
            "head_ref": "student/candidate",
            "head_sha": remote_head,
        },
    )

    StudentWorkspaceReconciler(workspace)((event,))

    assert (
        subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        == local_head
    )
    assert (workspace / "notes.txt").read_text() == "dirty but recoverable\n"


def test_controller_continues_a_conversation_recorded_before_restart(
    tmp_path: Path,
):
    conversation_id = UUID("00000000-0000-0000-0000-000000000004")
    ledger = ConversationStateLedger(tmp_path / "conversation-state.json")
    ledger.mark_success(conversation_id, "")
    turns = Turns()
    event = ControllerEvent(
        kind="training_monitor",
        dedupe_key="training:finished",
        payload={"conversation_id": str(conversation_id)},
    )
    controller = Controller(
        role="student",
        mailbox=Mailbox([(event,), ()]),
        turns=turns,
        conversation_id=conversation_id,
        full_prompt="programme",
        conversation_state=ConversationStateLedger(
            tmp_path / "conversation-state.json"
        ),
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    assert turns.calls[0][1] == conversation_id
    assert "programme" not in turns.calls[0][0]


def test_controller_publishes_a_hard_deadline_for_its_current_phase(
    tmp_path: Path,
):
    lease_path = tmp_path / "controller-lease.json"
    observed_leases = []

    class LeaseReadingTurns(Turns):
        def run(self, *args, **kwargs):
            observed_leases.append(WorkerLease.read(lease_path))
            return super().run(*args, **kwargs)

    event = ControllerEvent(
        kind="review_ready",
        dedupe_key="review:17:abc",
        payload={"number": 17},
    )
    controller = Controller(
        role="advisor",
        mailbox=Mailbox([(event,)]),
        turns=LeaseReadingTurns(),
        conversation_id=UUID("00000000-0000-0000-0000-000000000048"),
        full_prompt="programme",
        progress=ProgressLease(lease_path),
        operation_timeout_seconds=123,
        turn_timeout_seconds=456,
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    lease = observed_leases[0]
    assert lease.phase == "openhands-turn"
    assert lease.pid > 0


def test_changed_system_context_is_injected_once_without_rotating_uuid(
    tmp_path: Path,
):
    conversation_id = UUID("00000000-0000-0000-0000-000000000006")
    conversation_state = ConversationStateLedger(
        tmp_path / "conversation-state.json"
    )
    conversation_state.mark_success(conversation_id, "old harness and role")
    event = ControllerEvent(
        kind="review_ready",
        dedupe_key="review:17:new-role",
        payload={"number": 17},
    )
    next_event = ControllerEvent(
        kind="review_ready",
        dedupe_key="review:18:new-role",
        payload={"number": 18},
    )
    turns = Turns()
    controller = Controller(
        role="advisor",
        mailbox=Mailbox([(event,), (next_event,), ()]),
        turns=turns,
        conversation_id=conversation_id,
        full_prompt="programme",
        system_context="current harness and role",
        conversation_state=conversation_state,
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    assert turns.calls[0][1] == conversation_id
    assert "# Updated Senpai system context" in turns.calls[0][0]
    assert "current harness and role" in turns.calls[0][0]
    assert "# Updated Senpai system context" not in turns.calls[1][0]
    assert conversation_state.is_context_current(
        conversation_id,
        "current harness and role",
    )


def pull(*, labels, body="", feedback_urls=False):
    value = {
        "number": 17,
        "title": "Try bounded change",
        "html_url": "https://github.test/acme/widgets/pull/17",
        "updated_at": "2026-07-29T18:00:00Z",
        "body": body,
        "head": {"ref": "student/candidate", "sha": "a" * 40},
        "labels": [{"name": label} for label in labels],
    }
    if feedback_urls:
        value.update(
            {
                "url": "https://api.github.test/repos/acme/widgets/pulls/17",
                "comments_url": (
                    "https://api.github.test/repos/acme/widgets/issues/17/comments"
                ),
                "review_comments_url": (
                    "https://api.github.test/repos/acme/widgets/pulls/17/comments"
                ),
            }
        )
    return value


def feedback(
    feedback_id,
    body,
    *,
    author="morganmcg1",
    association="OWNER",
    user_type="User",
    created_at="2026-07-29T18:01:00Z",
    **extra,
):
    value = {
        "id": feedback_id,
        "html_url": f"https://github.test/comment/{feedback_id}",
        "body": body,
        "created_at": created_at,
        "author_association": association,
        "user": {"login": author, "type": user_type},
    }
    value.update(extra)
    return value


def feedback_responses(*, issue_comments=(), reviews=(), inline_comments=()):
    return {
        (
            "https://api.github.test/repos/acme/widgets/issues/17/comments"
            "?per_page=100"
        ): list(issue_comments),
        (
            "https://api.github.test/repos/acme/widgets/pulls/17/reviews"
            "?per_page=100"
        ): list(reviews),
        (
            "https://api.github.test/repos/acme/widgets/pulls/17/comments"
            "?per_page=100"
        ): list(inline_comments),
    }


def assigned_student_feedback_mailbox(
    monkeypatch,
    responses,
    *,
    status="status:wip",
    revision_id="revision-2",
    feedback_path=None,
    feedback_batch_events=8,
    feedback_batch_bytes=32_000,
    trusted_actor="morganmcg1",
):
    assignment = AssignmentRecord(
        repo="acme/widgets",
        assignment_id="assignment-17",
        revision_id=revision_id,
        student="student-1",
        base_ref="research",
        base_sha="b" * 40,
        head_ref="student/candidate",
        head_sha="a" * 40,
    )
    mailbox = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="student",
        advisor_branch="research",
        student_name="student-1",
        api_url="https://api.github.test",
        trusted_actor=trusted_actor,
        feedback_path=feedback_path,
        feedback_batch_events=feedback_batch_events,
        feedback_batch_bytes=feedback_batch_bytes,
    )
    assigned_pull = pull(
        labels=("research", "student:student-1", status),
        body=render_assignment_marker(assignment),
        feedback_urls=True,
    )
    calls = []

    def github_objects(url):
        calls.append(url)
        return responses[url]

    monkeypatch.setattr(mailbox, "_pulls", lambda: [assigned_pull])
    monkeypatch.setattr(mailbox, "_issues", list)
    monkeypatch.setattr(mailbox._github, "objects", github_objects)
    return mailbox, calls


def run_student_controller(tmp_path, mailbox, turns, *, reconcile=None):
    Controller(
        role="student",
        mailbox=mailbox,
        turns=turns,
        conversation_id=UUID("00000000-0000-0000-0000-000000000017"),
        full_prompt="programme",
        conversation_for_events=StudentConversationSelector(
            AssignmentConversationRegistry(tmp_path / "students.json")
        ),
        reconcile=reconcile,
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    ).run(max_cycles=1)


def test_github_mailbox_uses_pr_labels_as_the_cross_node_protocol(
    monkeypatch,
):
    advisor = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="advisor",
        advisor_branch="research",
        students=("student-1", "student-2"),
    )
    monkeypatch.setattr(
        advisor,
        "_pulls",
        lambda: [
            pull(
                labels=(
                    "research",
                    "student:student-1",
                    "status:review",
                )
            )
        ],
    )
    monkeypatch.setattr(advisor, "_issues", list)

    events = advisor.poll()

    assert {event.kind for event in events} == {
        "review_ready",
        "idle_student",
    }
    assert (
        next(event for event in events if event.kind == "review_ready").payload[
            "number"
        ]
        == 17
    )
    assert next(event for event in events if event.kind == "idle_student").payload == {
        "student": "student-2"
    }


def test_student_assignment_event_carries_durable_assignment_identity(
    monkeypatch,
):
    assignment = AssignmentRecord(
        repo="acme/widgets",
        assignment_id="assignment-17",
        revision_id="revision-2",
        student="student-1",
        base_ref="research",
        base_sha="b" * 40,
        head_ref="student/candidate",
        head_sha="a" * 40,
    )
    student = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="student",
        advisor_branch="research",
        student_name="student-1",
    )
    monkeypatch.setattr(
        student,
        "_pulls",
        lambda: [
            pull(
                labels=(
                    "research",
                    "student:student-1",
                    "status:wip",
                ),
                body=render_assignment_marker(assignment),
            )
        ],
    )
    monkeypatch.setattr(student, "_issues", list)

    event = student.poll()[0]

    assert event.kind == "student_assignment"
    assert event.payload["assignment_id"] == "assignment-17"
    assert event.payload["revision_id"] == "revision-2"


def test_review_ready_student_receives_only_trusted_feedback_from_every_surface(
    monkeypatch,
    tmp_path: Path,
):
    responses = feedback_responses(
        issue_comments=[
            feedback(101, "Pause after the current arm."),
            feedback(
                102,
                "Follow up on the current baseline.",
                created_at="2026-07-29T18:04:00Z",
            ),
            feedback(
                103,
                "<!-- senpai-revision:v1 {} -->\n\nAutomated revision.",
            ),
            feedback(
                104,
                "Untrusted advice.",
                author="mallory",
                association="NONE",
            ),
            feedback(105, "Automation.", author="ci-bot", user_type="Bot"),
        ],
        reviews=[
            feedback(
                201,
                "Please preserve the control.",
                author="ada",
                association="MEMBER",
                submitted_at="2026-07-29T18:02:00Z",
                state="CHANGES_REQUESTED",
            ),
            feedback(
                202,
                "Unsubmitted draft.",
                submitted_at=None,
                state="PENDING",
            ),
        ],
        inline_comments=[
            feedback(
                301,
                "This branch needs the current default.",
                author="grace",
                association="COLLABORATOR",
                created_at="2026-07-29T18:03:00Z",
                pull_request_review_id=201,
                path="train.py",
                line=42,
            ),
            feedback(
                302,
                "Private draft inline comment.",
                created_at="2026-07-29T18:03:30Z",
                pull_request_review_id=202,
                path="train.py",
                line=43,
            ),
        ],
    )
    student, calls = assigned_student_feedback_mailbox(
        monkeypatch,
        responses,
        status="status:review",
        trusted_actor="MorganMcG1",
    )

    events = student.poll()
    batches = StudentConversationSelector(
        AssignmentConversationRegistry(tmp_path / "students.json")
    )(events)

    assert [event.dedupe_key for event in events] == [
        "student_pr_feedback:issue_comment:17:101",
        "student_pr_feedback:review:17:201",
        "student_pr_feedback:inline_comment:17:301",
        "student_pr_feedback:issue_comment:17:102",
    ]
    assert calls == list(responses)
    assert events[0].payload == {
        "number": 17,
        "pr_url": "https://github.test/acme/widgets/pull/17",
        "feedback_url": "https://github.test/comment/101",
        "feedback_id": 101,
        "feedback_type": "issue_comment",
        "assignment_id": "assignment-17",
        "revision_id": "revision-2",
        "author": "morganmcg1",
        "author_association": "OWNER",
        "message": "Pause after the current arm.",
        "created_at": "2026-07-29T18:01:00Z",
    }
    assert events[1].payload["state"] == "CHANGES_REQUESTED"
    assert events[2].payload["path"] == "train.py"
    assert events[2].payload["line"] == 42
    assert len(batches) == 1
    assert batches[0].conversation_id == AssignmentConversationRegistry(
        tmp_path / "students.json"
    ).for_assignment("assignment-17", "revision-2")


def test_feedback_ack_requires_success_and_survives_restart_and_revision(
    monkeypatch,
    tmp_path: Path,
):
    ack_path = tmp_path / "github-feedback.json"
    feedback_key = "student_pr_feedback:issue_comment:17:131"
    responses = feedback_responses(
        issue_comments=[feedback(131, "Durable feedback.")]
    )

    failed, _ = assigned_student_feedback_mailbox(
        monkeypatch,
        responses,
        feedback_path=ack_path,
    )
    run_student_controller(tmp_path, failed, SequencedTurns([1]))
    assert json.loads(ack_path.read_text(encoding="utf-8"))[feedback_key] == {
        "assignment_id": "assignment-17",
        "revision_id": "revision-2",
        "acknowledged": False,
    }

    revised, _ = assigned_student_feedback_mailbox(
        monkeypatch,
        responses,
        revision_id="revision-3",
        feedback_path=ack_path,
    )
    revised_events = revised.poll()
    assert [event.kind for event in revised_events] == ["student_pr_feedback"]
    assert revised_events[0].payload["revision_id"] == "revision-2"
    turns = Turns()
    reconciled = []
    run_student_controller(
        tmp_path,
        revised,
        turns,
        reconcile=lambda events: reconciled.append(
            tuple(event.kind for event in events)
        ),
    )
    assert [
        tuple(sorted(key.rsplit(":", 1)[0] for key in event_keys))
        for _, _, event_keys in turns.calls
    ] == [
        ("student_pr_feedback:issue_comment:17",),
        ("student_assignment:assignment-17",),
    ]
    assert reconciled == [
        ("student_pr_feedback",),
        ("student_assignment",),
    ]
    assert json.loads(ack_path.read_text(encoding="utf-8"))[feedback_key][
        "acknowledged"
    ] is True

    restarted, _ = assigned_student_feedback_mailbox(
        monkeypatch,
        responses,
        revision_id="revision-4",
        feedback_path=ack_path,
    )
    restarted_events = restarted.poll()
    assert [event.kind for event in restarted_events] == ["student_assignment"]
    assert restarted_events[0].payload["revision_id"] == "revision-4"


def test_controller_drains_feedback_batches_oldest_first_after_success(
    monkeypatch,
    tmp_path: Path,
):
    comments = [
        feedback(
            140 + index,
            f"feedback-{index}",
            created_at=f"2026-07-29T18:01:{index:02d}Z",
        )
        for index in range(5)
    ]
    mailbox, _ = assigned_student_feedback_mailbox(
        monkeypatch,
        feedback_responses(issue_comments=comments),
        feedback_path=tmp_path / "feedback.json",
        feedback_batch_events=2,
        feedback_batch_bytes=100_000,
    )
    turns = Turns()
    run_student_controller(tmp_path, mailbox, turns)
    feedback_batches = [
        sorted(
            int(key.rsplit(":", 1)[1])
            for key in event_keys
            if key.startswith("student_pr_feedback:")
        )
        for _, _, event_keys in turns.calls
    ]

    assert feedback_batches == [[140, 141], [142, 143], [144]]
    assert not any(
        event.kind == "student_pr_feedback" for event in mailbox.poll()
    )


def test_feedback_batch_uses_rendered_prompt_byte_limit(monkeypatch):
    comments = [
        feedback(
            150 + index,
            "x" * 1_000,
            created_at=f"2026-07-29T18:02:{index:02d}Z",
        )
        for index in range(2)
    ]
    responses = feedback_responses(issue_comments=comments)
    probe, _ = assigned_student_feedback_mailbox(monkeypatch, responses)
    probe_events = [
        event for event in probe.poll() if event.kind == "student_pr_feedback"
    ]
    byte_limit = len(probe_events[0].to_prompt().encode())
    bounded, _ = assigned_student_feedback_mailbox(
        monkeypatch,
        responses,
        feedback_batch_events=8,
        feedback_batch_bytes=byte_limit,
    )

    batch = [
        event for event in bounded.poll() if event.kind == "student_pr_feedback"
    ]
    rendered_bytes = len("\n\n".join(event.to_prompt() for event in batch).encode())

    assert len(batch) == 1
    assert rendered_bytes <= byte_limit


def test_long_feedback_preserves_head_and_tail_and_points_to_full_message(
    monkeypatch,
):
    message = "ACTION: stop after this run.\n" + "x" * 5_000 + "\nTAIL: thanks"
    mailbox, _ = assigned_student_feedback_mailbox(
        monkeypatch,
        feedback_responses(issue_comments=[feedback(160, message)]),
    )

    event = next(
        event for event in mailbox.poll() if event.kind == "student_pr_feedback"
    )

    assert str(event.payload["message"]).startswith("ACTION: stop after this run.")
    assert str(event.payload["message"]).endswith("TAIL: thanks")
    assert len(str(event.payload["message"]).encode()) <= 4_000
    assert event.payload["message_truncated"] is True
    assert event.payload["full_message_instruction"] == (
        "Open feedback_url to read the omitted text."
    )


def test_student_pr_feedback_groups_with_monitor_wake_on_assignment_uuid(
    tmp_path: Path,
):
    registry = AssignmentConversationRegistry(tmp_path / "students.json")
    conversation_id = registry.for_assignment("assignment-17", "revision-2")
    events = (
        ControllerEvent(
            kind="student_assignment",
            dedupe_key="student_assignment:assignment-17:revision-2",
            payload={
                "assignment_id": "assignment-17",
                "revision_id": "revision-2",
            },
        ),
        ControllerEvent(
            kind="student_pr_feedback",
            dedupe_key="student_pr_feedback:issue_comment:17:101",
            payload={
                "assignment_id": "assignment-17",
                "revision_id": "revision-2",
            },
        ),
        ControllerEvent(
            kind="training_monitor",
            dedupe_key="training_monitor:run-1:finished",
            payload={"conversation_id": str(conversation_id)},
        ),
    )

    batches = StudentConversationSelector(registry)(events)

    assert len(batches) == 1
    assert batches[0].conversation_id == conversation_id
    assert batches[0].events == events


def test_malformed_assignment_does_not_suppress_human_messages(
    monkeypatch,
):
    student = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="student",
        advisor_branch="research",
        student_name="student-1",
        trusted_actor="senpai-bot",
    )
    malformed = pull(
        labels=("research", "student:student-1", "status:wip"),
        body="<!-- senpai-assignment:v1 not-json -->",
    )
    issue = {
        "id": 700,
        "number": 23,
        "title": "Please investigate",
        "html_url": "https://github.test/acme/widgets/issues/23",
        "created_at": "2026-07-29T18:00:00Z",
        "body": "The assignment marker looks broken.",
        "user": {"login": "ada"},
        "labels": [{"name": "human"}, {"name": "student:student-1"}],
    }
    monkeypatch.setattr(student, "_pulls", lambda: [malformed])
    monkeypatch.setattr(student, "_issues", lambda: [issue])
    monkeypatch.setattr(student, "_issue_comments", lambda _issue: [])

    events = student.poll()

    assert [event.kind for event in events] == [
        "malformed_assignment",
        "human_issue",
    ]
    error = events[0]
    assert error.dedupe_key == f"malformed_assignment:17:{'a' * 40}"
    assert error.payload["number"] == 17
    assert "malformed" in str(error.payload["error"]).lower()
    assert events[1].payload["human_message_id"] == 700


def test_human_issue_event_tracks_the_exact_latest_human_message(
    monkeypatch,
):
    mailbox = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="advisor",
        advisor_branch="research",
        trusted_actor="senpai-bot",
    )
    issue = {
        "id": 700,
        "number": 23,
        "title": "Change direction",
        "html_url": "https://github.test/acme/widgets/issues/23",
        "updated_at": "2026-07-29T18:10:00Z",
        "created_at": "2026-07-29T18:00:00Z",
        "body": "Start with the cheaper baseline.",
        "user": {"login": "ada"},
        "labels": [{"name": "human"}, {"name": "team"}],
    }
    monkeypatch.setattr(mailbox, "_pulls", list)
    monkeypatch.setattr(mailbox, "_issues", lambda: [issue])
    monkeypatch.setattr(
        mailbox,
        "_issue_comments",
        lambda _issue: [
            {
                "id": 701,
                "body": "ADVISOR: acknowledged",
                "created_at": "2026-07-29T18:05:00Z",
                "user": {"login": "SENPAI-BOT"},
            },
            {
                "id": 702,
                "body": "Also compare memory.",
                "created_at": "2026-07-29T18:10:00Z",
                "user": {"login": "ada"},
            },
        ],
    )

    event = mailbox.poll()[0]

    assert event.kind == "human_issue"
    assert event.dedupe_key == "human_issue:23:702"
    assert event.payload["human_message_id"] == 702
    assert event.payload["author"] == "ada"
    assert event.payload["message"] == "Also compare memory."


def test_human_issue_polling_can_be_disabled_for_an_isolated_launch(
    monkeypatch,
):
    mailbox = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="advisor",
        advisor_branch="research",
        trusted_actor="senpai-bot",
        human_issues_enabled=False,
    )
    monkeypatch.setattr(mailbox, "_pulls", list)
    monkeypatch.setattr(
        mailbox,
        "_issues",
        lambda: [
            {
                "id": 700,
                "number": 23,
                "title": "Out of scope",
                "html_url": "https://github.test/acme/widgets/issues/23",
                "updated_at": "2026-07-29T18:10:00Z",
                "created_at": "2026-07-29T18:00:00Z",
                "body": "Do not deliver this.",
                "user": {"login": "ada"},
                "labels": [{"name": "human"}, {"name": "team"}],
            }
        ],
    )

    assert mailbox.poll() == ()
