import json
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from senpai_agent.state import ConversationStateLedger


def test_conversation_state_migrates_legacy_ledgers_atomically(
    tmp_path: Path,
) -> None:
    current_id = UUID("00000000-0000-0000-0000-000000000001")
    started_without_context_id = UUID("00000000-0000-0000-0000-000000000002")
    context = "current harness and role"
    digest = sha256(context.encode()).hexdigest()
    (tmp_path / "started-conversations.json").write_text(
        json.dumps([str(current_id), str(started_without_context_id)]),
        encoding="utf-8",
    )
    (tmp_path / "system-context-revisions.json").write_text(
        json.dumps({str(current_id): digest}),
        encoding="utf-8",
    )

    ledger = ConversationStateLedger(tmp_path / "conversation-state.json")

    assert ledger.has_started(current_id)
    assert ledger.has_started(started_without_context_id)
    assert ledger.is_context_current(current_id, context)
    assert not ledger.is_context_current(started_without_context_id, context)
    assert json.loads((tmp_path / "conversation-state.json").read_text()) == {
        str(current_id): digest,
        str(started_without_context_id): "",
    }


def test_conversation_state_migration_runs_only_once(tmp_path: Path) -> None:
    migrated_id = UUID("00000000-0000-0000-0000-000000000003")
    later_legacy_id = UUID("00000000-0000-0000-0000-000000000004")
    started_path = tmp_path / "started-conversations.json"
    started_path.write_text(json.dumps([str(migrated_id)]), encoding="utf-8")
    state_path = tmp_path / "conversation-state.json"

    ConversationStateLedger(state_path)
    started_path.write_text(
        json.dumps([str(migrated_id), str(later_legacy_id)]),
        encoding="utf-8",
    )

    ledger = ConversationStateLedger(state_path)

    assert ledger.has_started(migrated_id)
    assert not ledger.has_started(later_legacy_id)
