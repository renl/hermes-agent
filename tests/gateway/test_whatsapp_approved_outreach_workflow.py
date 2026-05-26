from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource
from gateway.whatsapp_message_store import append_whatsapp_record
from gateway.whatsapp_approved_outreach import (
    WHATSAPP_DEFAULT_COMMUNICATION_SKILL_NAME,
    WHATSAPP_COLD_START_VALIDATION_MODE_NON_DESTRUCTIVE_ISOLATION,
    WHATSAPP_COLD_START_VALIDATION_CONTINUITY_SCOPE_KIND,
    WHATSAPP_PRODUCTION_CONTINUITY_SCOPE_KIND,
    build_cli_chat_whatsapp_outreach_requests,
    bind_whatsapp_outreach_plan_to_cron_job,
    execute_whatsapp_approved_outreach,
    format_whatsapp_approved_outreach_result,
    load_whatsapp_outreach_run_records,
    load_whatsapp_outreach_state,
    prepare_whatsapp_cold_start_validation,
    _write_whatsapp_outreach_state,
)


def _make_source(
    *,
    platform: Platform = Platform.WHATSAPP,
    user_id: str = "15550000001",
    chat_id: str = "15551230000@s.whatsapp.net",
    chat_type: str = "dm",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="Founder",
        chat_type=chat_type,
    )


def _make_event(
    text: str,
    *,
    participant_role: str | None = "owner_operator",
    command_authority_scope: str | None = "owner_only",
    platform: Platform = Platform.WHATSAPP,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(platform=platform),
        message_id="evt-1",
        participant_role=participant_role,
        message_classification=(
            "command_capable"
            if participant_role == "owner_operator"
            else "conversational_only"
        ),
        command_authority_scope=command_authority_scope,
    )


def _make_runner(tmp_path):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(
                enabled=True,
                extra={"allow_admin_from": ["15550000001"]},
            )
        }
    )
    runner.adapters = {Platform.WHATSAPP: MagicMock()}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:whatsapp:dm:15551230000",
        session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        platform=Platform.WHATSAPP,
        chat_type="dm",
        total_tokens=0,
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._session_sources = {}
    runner._session_run_generation = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


def _append_record(
    base_dir,
    *,
    record_id: str,
    text: str,
    participant_role: str,
    message_id: str,
    effective_event_at: str,
) -> None:
    append_whatsapp_record(
        {
            "record_kind": "conversation_record",
            "record_id": record_id,
            "conversation_key": "whatsapp:dm:15551230000",
            "destination_key": "whatsapp:dm:15551230000",
            "destination_context_type": "direct_message",
            "destination_chat_id": "15551230000@s.whatsapp.net",
            "destination_target_id": "15551230000",
            "group_chat_id": None,
            "dm_counterparty_id": "15551230000",
            "direction": "outbound" if participant_role == "agent" else "inbound",
            "effective_event_at_utc": effective_event_at,
            "record_sequence": int(record_id.split("-")[-1]),
            "participant_role": participant_role,
            "message_classification": (
                "command_capable"
                if participant_role == "owner_operator"
                else "conversational_only"
            ),
            "command_authority_scope": (
                "owner_only" if participant_role == "owner_operator" else "none"
            ),
            "sender_id": "agent" if participant_role == "agent" else "15551230000",
            "sender_name": "Hermes" if participant_role == "agent" else "Vendor",
            "text": text,
            "message_id": message_id,
            "media_types": [],
        },
        effective_event_at=datetime.fromisoformat(
            effective_event_at.replace("Z", "+00:00")
        ),
        base_dir=base_dir,
    )


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_sends_one_bounded_follow_up(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock(
        return_value=SendResult(
            success=True,
            message_id="bridge-msg-9",
            raw_response={
                "dispatch_group_id": "dispatch-123",
                "messageId": "bridge-msg-9",
            },
        )
    )
    monkeypatch.setattr(
        "gateway.whatsapp_approved_outreach._draft_whatsapp_message_via_skill",
        lambda **_kwargs: pytest.fail("skill drafting should be bypassed"),
    )

    result = await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."'
        )
    )

    runner.adapters[Platform.WHATSAPP].send.assert_awaited_once_with(
        "15551230000@s.whatsapp.net",
        "Following up on the revised quote.",
    )
    assert "WhatsApp approved outreach" in result
    assert "Status: ready" in result
    assert "trigger_source: owner_instruction" in result
    assert "run_status: completed" in result
    assert "message_generation_mode: operator_text_override" in result
    assert "draft_outcome: send_message" in result
    assert "execution_status: sent" in result
    assert "dispatch_group_id: dispatch-123" in result
    assert "message_id: bridge-msg-9" in result

    run_records = load_whatsapp_outreach_run_records()
    assert len(run_records) == 1
    persisted = run_records[0]
    assert (
        persisted["plan"]["approved_targets"][0]["max_outbound_messages_per_run"] == 1
    )
    assert persisted["run"]["trigger_source"] == "owner_instruction"
    assert persisted["run"]["target_executions"][0]["execution_status"] == "sent"

    outreach_state = load_whatsapp_outreach_state()
    assert outreach_state["schema_version"] == 1
    assert len(outreach_state["plans"]) == 1
    assert len(outreach_state["plan_targets"]) == 1
    assert len(outreach_state["runs"]) == 1
    assert len(outreach_state["target_executions"]) == 1
    assert len(outreach_state["reports"]) == 1

    plan_row = outreach_state["plans"][0]
    target_row = outreach_state["plan_targets"][0]
    run_row = outreach_state["runs"][0]
    execution_row = outreach_state["target_executions"][0]
    report_row = outreach_state["reports"][0]

    assert plan_row["plan_status"] == "active"
    assert (
        plan_row["communication_skill_name"]
        == WHATSAPP_DEFAULT_COMMUNICATION_SKILL_NAME
    )
    assert plan_row["operator_objective"] == "request the revised quote"
    assert target_row["plan_id"] == plan_row["plan_id"]
    assert target_row["conversation_key"] == "whatsapp:dm:15551230000"
    assert target_row["destination_key"] == "whatsapp:dm:15551230000"
    assert target_row["dm_counterparty_id"] == "15551230000"
    assert target_row["max_outbound_messages_per_run"] == 1
    assert run_row["plan_id"] == plan_row["plan_id"]
    assert run_row["run_status"] == "completed"
    assert run_row["trigger_source"] == "owner_instruction"
    assert run_row["target_count"] == 1
    assert run_row["completed_target_count"] == 1
    assert run_row["failed_target_count"] == 0
    assert run_row["report_id"] == report_row["report_id"]
    assert execution_row["run_id"] == run_row["run_id"]
    assert execution_row["plan_target_id"] == target_row["plan_target_id"]
    assert execution_row["execution_status"] == "sent"
    assert execution_row["message_generation_mode"] == "operator_text_override"
    assert execution_row["draft_outcome"] == "send_message"
    assert (
        execution_row["outbound_message_text"] == "Following up on the revised quote."
    )
    assert execution_row["resolved_conversation_key"] == "whatsapp:dm:15551230000"
    assert execution_row["resolved_destination_key"] == "whatsapp:dm:15551230000"
    assert execution_row["resolved_destination_chat_id"] == "15551230000@s.whatsapp.net"
    assert execution_row["dispatch_group_id"] == "dispatch-123"
    assert execution_row["message_id"] == "bridge-msg-9"
    assert report_row["plan_id"] == plan_row["plan_id"]
    assert report_row["run_id"] == run_row["run_id"]
    assert report_row["report_status"] == "ready"


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_loads_skill_and_sends_generated_message(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    loader_calls: list[str] = []
    agent_prompts: list[str] = []

    def _fake_load_skill_payload(skill_identifier: str, task_id: str | None = None):
        loader_calls.append(skill_identifier)
        return (
            {
                "success": True,
                "name": "whatsapp-communications",
                "content": "Draft one bounded WhatsApp decision.",
                "raw_content": "---\nname: whatsapp-communications\n---\n# Skill\n",
            },
            tmp_path / "skills" / "communication" / "whatsapp-communications",
            "whatsapp-communications",
        )

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def run_conversation(
            self,
            user_message,
            system_message=None,
            conversation_history=None,
            task_id=None,
        ):
            agent_prompts.append(user_message)
            assert system_message is not None
            assert conversation_history
            assert "whatsapp-communications" in conversation_history[0]["content"]
            return {
                "final_response": (
                    '{"draft_outcome":"send_message",'
                    '"outbound_message_text":"Skill-generated follow-up.",'
                    '"handoff_reason":null}'
                )
            }

    monkeypatch.setattr(
        "gateway.whatsapp_approved_outreach._load_skill_payload",
        _fake_load_skill_payload,
    )
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock(
        return_value=SendResult(
            success=True,
            message_id="bridge-msg-skill",
            raw_response={"dispatch_group_id": "dispatch-skill"},
        )
    )

    result = await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote"'
        )
    )

    assert loader_calls == ["whatsapp-communications"]
    assert agent_prompts
    assert '"communication_skill_name": "whatsapp-communications"' in agent_prompts[0]
    assert '"message_text_override": null' in agent_prompts[0]
    runner.adapters[Platform.WHATSAPP].send.assert_awaited_once_with(
        "15551230000@s.whatsapp.net",
        "Skill-generated follow-up.",
    )
    assert "message_generation_mode: skill_generated" in result
    assert "draft_outcome: send_message" in result
    assert "outbound_message_text: Skill-generated follow-up." in result
    assert "execution_status: sent" in result

    execution_row = load_whatsapp_outreach_state()["target_executions"][0]
    assert execution_row["communication_skill_name"] == "whatsapp-communications"
    assert execution_row["message_generation_mode"] == "skill_generated"
    assert execution_row["communication_stage"] == "opening_outreach"
    assert execution_row["draft_outcome"] == "send_message"
    assert execution_row["outbound_message_text"] == "Skill-generated follow-up."
    assert execution_row["handoff_reason"] is None


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_reuses_durable_plan_and_target_state(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock(
        return_value=SendResult(
            success=True,
            message_id="bridge-msg-10",
            raw_response={
                "dispatch_group_id": "dispatch-124",
                "messageId": "bridge-msg-10",
            },
        )
    )

    instruction = (
        "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
        'operator_objective="request the revised quote" '
        'message_text="Following up on the revised quote."'
    )
    await runner._handle_message(_make_event(instruction, platform=Platform.WHATSAPP))
    await runner._handle_message(_make_event(instruction, platform=Platform.WHATSAPP))

    run_records = load_whatsapp_outreach_run_records()
    assert len(run_records) == 2
    first_plan_id = run_records[0]["plan"]["plan_id"]
    second_plan_id = run_records[1]["plan"]["plan_id"]
    first_target_id = run_records[0]["plan"]["approved_targets"][0]["plan_target_id"]
    second_target_id = run_records[1]["plan"]["approved_targets"][0]["plan_target_id"]

    assert first_plan_id == second_plan_id
    assert first_target_id == second_target_id
    assert run_records[0]["run"]["run_id"] != run_records[1]["run"]["run_id"]
    assert (
        run_records[0]["run"]["target_executions"][0]["target_execution_id"]
        != run_records[1]["run"]["target_executions"][0]["target_execution_id"]
    )

    outreach_state = load_whatsapp_outreach_state()
    assert len(outreach_state["plans"]) == 1
    assert len(outreach_state["plan_targets"]) == 1
    assert len(outreach_state["runs"]) == 2
    assert len(outreach_state["target_executions"]) == 2
    assert len(outreach_state["reports"]) == 2


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_can_complete_without_send(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "gateway.whatsapp_approved_outreach._draft_whatsapp_message_via_skill",
        lambda **_kwargs: (
            {
                "communication_skill_name": "whatsapp-communications",
                "communication_stage": "opening_outreach",
                "operator_objective": "check whether a follow-up is needed",
                "resolved_target": {},
                "history_rows": [],
                "message_text_override": None,
                "trigger_source": "owner_instruction",
            },
            {
                "draft_outcome": "no_send_required",
                "outbound_message_text": None,
                "handoff_reason": None,
            },
        ),
    )

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock()

    result = await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="check whether a follow-up is needed"'
        )
    )

    runner.adapters[Platform.WHATSAPP].send.assert_not_awaited()
    assert "Status: ready" in result
    assert "run_status: completed" in result
    assert "message_generation_mode: skill_generated" in result
    assert "draft_outcome: no_send_required" in result
    assert "execution_status: no_send_required" in result
    assert "history_record_count: 1" in result


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_can_require_handoff_from_skill(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "gateway.whatsapp_approved_outreach._draft_whatsapp_message_via_skill",
        lambda **_kwargs: (
            {
                "communication_skill_name": "whatsapp-communications",
                "communication_stage": "opening_outreach",
                "operator_objective": "request the revised quote",
                "resolved_target": {},
                "history_rows": [],
                "message_text_override": None,
                "trigger_source": "owner_instruction",
            },
            {
                "draft_outcome": "handoff_required",
                "outbound_message_text": None,
                "handoff_reason": "pricing requires owner review",
            },
        ),
    )

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock()

    result = await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote"'
        )
    )

    runner.adapters[Platform.WHATSAPP].send.assert_not_awaited()
    assert "message_generation_mode: skill_generated" in result
    assert "draft_outcome: handoff_required" in result
    assert "execution_status: handoff_required" in result
    assert "handoff_reason: pricing requires owner review" in result

    execution_row = load_whatsapp_outreach_state()["target_executions"][0]
    assert execution_row["draft_outcome"] == "handoff_required"
    assert execution_row["outbound_message_text"] is None
    assert execution_row["handoff_reason"] == "pricing requires owner review"


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_fails_closed_for_ambiguous_target(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock()

    result = await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            "dm_counterparty_id=15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."'
        )
    )

    runner.adapters[Platform.WHATSAPP].send.assert_not_awaited()
    assert "Status: invalid_request" in result


def test_cli_chat_instruction_normalizes_into_canonical_outreach_request():
    operator_instruction, run_request = build_cli_chat_whatsapp_outreach_requests(
        (
            "whatsapp outreach approved_destination_chat_id=15551230000@s.whatsapp.net "
            'operator_objective="Request the first quote" '
            'message_text="Hello from Hermes."'
        ),
        session_id="sess-cli-1",
    )

    assert (
        operator_instruction["approved_destination_chat_id"]
        == "15551230000@s.whatsapp.net"
    )
    assert operator_instruction["trigger_source"] == "owner_instruction"
    assert operator_instruction["trigger_reference_id"] == "sess-cli-1"
    assert operator_instruction["operator_ingress_surface"] == "cli_chat"
    assert run_request == operator_instruction


def test_prepare_cold_start_validation_creates_non_destructive_exact_target_scope(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _write_whatsapp_outreach_state({
        "schema_version": 1,
        "plans": [
            {
                "plan_id": "waplan-1",
                "plan_status": "active",
                "trigger_mode": "instruction_only",
                "communication_skill_name": WHATSAPP_DEFAULT_COMMUNICATION_SKILL_NAME,
                "operator_objective": "request quote",
                "approved_by_principal": "owner_operator",
            }
        ],
        "plan_targets": [
            {
                "plan_target_id": "watarget-1",
                "plan_id": "waplan-1",
                "target_status": "active",
                "conversation_key": "whatsapp:dm:15551230000",
                "destination_key": "whatsapp:dm:15551230000",
                "group_chat_id": None,
                "dm_counterparty_id": "15551230000",
                "approved_destination_chat_id": "15551230000@s.whatsapp.net",
                "target_resolution_source": "preserved_history",
                "continuity_mode": "preserved_thread_follow_up",
            }
        ],
        "continuity_scopes": [
            {
                "continuity_scope_id": "scope-production-1",
                "plan_target_id": "watarget-1",
                "continuity_scope_kind": WHATSAPP_PRODUCTION_CONTINUITY_SCOPE_KIND,
                "cold_start_validation_mode": "none",
                "scope_status": "active",
                "target_destination_key": "whatsapp:dm:15551230000",
                "target_dm_counterparty_id": "15551230000",
                "approved_destination_chat_id_snapshot": "15551230000@s.whatsapp.net",
                "created_by_principal": "owner_operator",
                "created_at_utc": "2024-06-02T09:00:00Z",
                "operator_reason": "existing production continuity",
            }
        ],
        "runs": [],
        "target_executions": [],
        "reports": [],
    })

    result = prepare_whatsapp_cold_start_validation(
        {
            "destination_key": "whatsapp:dm:15551230000",
            "cold_start_validation_mode": WHATSAPP_COLD_START_VALIDATION_MODE_NON_DESTRUCTIVE_ISOLATION,
            "operator_reason": "prepare repeatable strict cold-start validation",
        },
        authorized=True,
        created_by_principal="owner_operator",
    )

    assert result["workflow_status"] == "prepared"
    continuity_scope = result["continuity_scope"]
    assert continuity_scope["continuity_scope_kind"] == (
        WHATSAPP_COLD_START_VALIDATION_CONTINUITY_SCOPE_KIND
    )
    assert continuity_scope["cold_start_validation_mode"] == (
        WHATSAPP_COLD_START_VALIDATION_MODE_NON_DESTRUCTIVE_ISOLATION
    )
    assert continuity_scope["target_destination_key"] == "whatsapp:dm:15551230000"
    assert continuity_scope["target_dm_counterparty_id"] == "15551230000"
    assert continuity_scope["approved_destination_chat_id_snapshot"] == (
        "15551230000@s.whatsapp.net"
    )
    assert continuity_scope["created_by_principal"] == "owner_operator"
    assert continuity_scope["operator_reason"] == (
        "prepare repeatable strict cold-start validation"
    )
    assert continuity_scope["continuity_scope_id"]

    state = load_whatsapp_outreach_state()
    assert state["plan_targets"][0].get("default_continuity_scope_id") == (
        "scope-production-1"
    )
    assert len(state["continuity_scopes"]) == 2
    persisted_scope = state["continuity_scopes"][-1]
    assert (
        persisted_scope["continuity_scope_id"]
        == continuity_scope["continuity_scope_id"]
    )
    assert persisted_scope["scope_status"] == "prepared"


def test_prepare_cold_start_validation_rejects_group_and_ambiguous_targets(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _write_whatsapp_outreach_state({
        "schema_version": 1,
        "plans": [
            {
                "plan_id": "waplan-1",
                "plan_status": "active",
                "trigger_mode": "instruction_only",
                "communication_skill_name": WHATSAPP_DEFAULT_COMMUNICATION_SKILL_NAME,
                "operator_objective": "request quote",
                "approved_by_principal": "owner_operator",
            }
        ],
        "plan_targets": [
            {
                "plan_target_id": "watarget-group",
                "plan_id": "waplan-1",
                "target_status": "active",
                "conversation_key": "whatsapp:group:group-123@g.us",
                "destination_key": "whatsapp:group:group-123@g.us",
                "group_chat_id": "group-123@g.us",
                "dm_counterparty_id": None,
                "target_resolution_source": "preserved_history",
                "continuity_mode": "preserved_thread_follow_up",
            },
            {
                "plan_target_id": "watarget-dm-a",
                "plan_id": "waplan-1",
                "target_status": "active",
                "conversation_key": "whatsapp:dm:15551230000",
                "destination_key": "whatsapp:dm:15551230000",
                "group_chat_id": None,
                "dm_counterparty_id": "15551230000",
                "target_resolution_source": "preserved_history",
                "continuity_mode": "preserved_thread_follow_up",
            },
            {
                "plan_target_id": "watarget-dm-b",
                "plan_id": "waplan-1",
                "target_status": "active",
                "conversation_key": "whatsapp:dm:15551230000",
                "destination_key": "whatsapp:dm:15551230000",
                "group_chat_id": None,
                "dm_counterparty_id": "15551230000",
                "target_resolution_source": "preserved_history",
                "continuity_mode": "preserved_thread_follow_up",
            },
        ],
        "continuity_scopes": [],
        "runs": [],
        "target_executions": [],
        "reports": [],
    })

    group_result = prepare_whatsapp_cold_start_validation(
        {
            "group_chat_id": "group-123@g.us",
            "cold_start_validation_mode": WHATSAPP_COLD_START_VALIDATION_MODE_NON_DESTRUCTIVE_ISOLATION,
            "operator_reason": "prepare validation",
        },
        authorized=True,
        created_by_principal="owner_operator",
    )
    assert group_result["workflow_status"] == "invalid_request"
    assert (
        group_result["reason"]
        == "group targets are unsupported for cold-start validation"
    )

    ambiguous_result = prepare_whatsapp_cold_start_validation(
        {
            "destination_key": "whatsapp:dm:15551230000",
            "cold_start_validation_mode": WHATSAPP_COLD_START_VALIDATION_MODE_NON_DESTRUCTIVE_ISOLATION,
            "operator_reason": "prepare validation",
        },
        authorized=True,
        created_by_principal="owner_operator",
    )
    assert ambiguous_result["workflow_status"] == "blocked"
    assert ambiguous_result["reason"] == "exact approved DM target is ambiguous"


@pytest.mark.asyncio
async def test_gateway_runner_prepare_surface_returns_validation_scope_details(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_whatsapp_outreach_state({
        "schema_version": 1,
        "plans": [
            {
                "plan_id": "waplan-1",
                "plan_status": "active",
                "trigger_mode": "instruction_only",
                "communication_skill_name": WHATSAPP_DEFAULT_COMMUNICATION_SKILL_NAME,
                "operator_objective": "request quote",
                "approved_by_principal": "owner_operator",
            }
        ],
        "plan_targets": [
            {
                "plan_target_id": "watarget-1",
                "plan_id": "waplan-1",
                "target_status": "active",
                "conversation_key": "whatsapp:dm:15551230000",
                "destination_key": "whatsapp:dm:15551230000",
                "group_chat_id": None,
                "dm_counterparty_id": "15551230000",
                "approved_destination_chat_id": "15551230000@s.whatsapp.net",
                "target_resolution_source": "preserved_history",
                "continuity_mode": "preserved_thread_follow_up",
                "default_continuity_scope_id": "scope-production-1",
            }
        ],
        "continuity_scopes": [
            {
                "continuity_scope_id": "scope-production-1",
                "plan_target_id": "watarget-1",
                "continuity_scope_kind": WHATSAPP_PRODUCTION_CONTINUITY_SCOPE_KIND,
                "cold_start_validation_mode": "none",
                "scope_status": "active",
                "target_destination_key": "whatsapp:dm:15551230000",
                "target_dm_counterparty_id": "15551230000",
                "approved_destination_chat_id_snapshot": "15551230000@s.whatsapp.net",
                "created_by_principal": "owner_operator",
                "created_at_utc": "2024-06-02T09:00:00Z",
                "operator_reason": "existing production continuity",
            }
        ],
        "runs": [],
        "target_executions": [],
        "reports": [],
    })

    runner = _make_runner(tmp_path)
    rendered = await runner._handle_whatsapp_cold_start_validation_prepare(
        {
            "destination_key": "whatsapp:dm:15551230000",
            "cold_start_validation_mode": WHATSAPP_COLD_START_VALIDATION_MODE_NON_DESTRUCTIVE_ISOLATION,
            "operator_reason": "prepare validation from runner seam",
        },
        authorized=True,
        created_by_principal="owner_operator",
    )

    assert "WhatsApp cold-start validation" in rendered
    assert "Status: prepared" in rendered
    assert (
        f"continuity_scope_kind: {WHATSAPP_COLD_START_VALIDATION_CONTINUITY_SCOPE_KIND}"
        in rendered
    )
    assert (
        f"cold_start_validation_mode: {WHATSAPP_COLD_START_VALIDATION_MODE_NON_DESTRUCTIVE_ISOLATION}"
        in rendered
    )
    assert "target_dm_counterparty_id: 15551230000" in rendered


@pytest.mark.asyncio
async def test_cli_chat_cold_start_instruction_sends_one_first_dm_and_persists_honest_continuity(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock(
        return_value=SendResult(
            success=True,
            message_id="bridge-cold-start-1",
            raw_response={
                "dispatch_group_id": "dispatch-cold-start-1",
                "messageId": "bridge-cold-start-1",
            },
        )
    )

    operator_instruction, run_request = build_cli_chat_whatsapp_outreach_requests(
        (
            "whatsapp outreach approved_destination_chat_id=15551230000@s.whatsapp.net "
            'operator_objective="Request the first quote" '
            'message_text="Hello from Hermes."'
        ),
        session_id="sess-cli-cold-start",
    )

    result = await execute_whatsapp_approved_outreach(
        run_request,
        authorized=True,
        adapter=runner.adapters[Platform.WHATSAPP],
    )

    assert operator_instruction["operator_ingress_surface"] == "cli_chat"
    runner.adapters[Platform.WHATSAPP].send.assert_awaited_once_with(
        "15551230000@s.whatsapp.net",
        "Hello from Hermes.",
    )
    assert result["workflow_status"] == "ready"
    assert result["run"]["trigger_source"] == "owner_instruction"
    assert result["run"]["operator_ingress_surface"] == "cli_chat"
    assert (
        result["execution"]["target_resolution_source"]
        == "approved_destination_chat_id"
    )
    assert (
        result["execution"]["approved_destination_chat_id_snapshot"]
        == "15551230000@s.whatsapp.net"
    )
    assert result["execution"]["had_prior_preserved_thread"] is False
    assert result["execution"]["continuity_mode"] == "cold_start_first_send_completed"
    assert result["execution"]["communication_stage"] == "opening_outreach"
    assert result["execution"]["draft_outcome"] == "send_message"
    assert result["execution"]["execution_status"] == "sent"
    assert (
        result["execution"]["resolved_target"]["destination_chat_id"]
        == "15551230000@s.whatsapp.net"
    )
    assert (
        result["execution"]["resolved_target"]["destination_key"]
        == "whatsapp:dm:15551230000"
    )
    assert (
        result["execution"]["resolved_target"]["conversation_key"]
        == "whatsapp:dm:15551230000"
    )

    rendered = format_whatsapp_approved_outreach_result(result)
    assert "operator_ingress_surface: cli_chat" in rendered
    assert "target_resolution_source: approved_destination_chat_id" in rendered
    assert (
        "approved_destination_chat_id_snapshot: 15551230000@s.whatsapp.net" in rendered
    )
    assert "had_prior_preserved_thread: False" in rendered
    assert "continuity_mode: cold_start_first_send_completed" in rendered
    assert "report_continuity_mode: cold_start_first_send_completed" in rendered
    assert "report_had_prior_preserved_thread: False" in rendered

    outreach_state = load_whatsapp_outreach_state()
    plan_row = outreach_state["plans"][0]
    target_row = outreach_state["plan_targets"][0]
    run_row = outreach_state["runs"][0]
    execution_row = outreach_state["target_executions"][0]
    report_row = outreach_state["reports"][0]

    assert plan_row["operator_ingress_surface"] == "cli_chat"
    assert target_row["approved_destination_chat_id"] == "15551230000@s.whatsapp.net"
    assert target_row["target_resolution_source"] == "preserved_history"
    assert target_row["continuity_mode"] == "preserved_thread_follow_up"
    assert target_row["destination_key"] == "whatsapp:dm:15551230000"
    assert target_row["dm_counterparty_id"] == "15551230000"
    assert run_row["operator_ingress_surface"] == "cli_chat"
    assert execution_row["target_resolution_source"] == "approved_destination_chat_id"
    assert (
        execution_row["approved_destination_chat_id_snapshot"]
        == "15551230000@s.whatsapp.net"
    )
    assert execution_row["had_prior_preserved_thread"] is False
    assert execution_row["continuity_mode"] == "cold_start_first_send_completed"
    assert execution_row["history_record_count"] == 0
    assert (
        report_row["target_rows"][0]["continuity_mode"]
        == "cold_start_first_send_completed"
    )
    assert report_row["target_rows"][0]["had_prior_preserved_thread"] is False
    assert any(
        "without prior preserved thread context" in item
        for item in report_row["target_rows"][0]["uncertainties"]
    )


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_is_forbidden_for_external_party(tmp_path):
    runner = _make_runner(tmp_path)

    result = await runner._handle_whatsapp_approved_outreach_instruction(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."',
            participant_role="external_party",
            command_authority_scope="none",
        )
    )

    assert "Status: forbidden" in result


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_surfaces_send_failure(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock(
        return_value=SendResult(success=False, error="bridge down")
    )

    result = await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."'
        )
    )

    assert "Status: ready" in result
    assert "run_status: failed" in result
    assert "execution_status: send_failed" in result
    assert "last_error: bridge down" in result


@pytest.mark.asyncio
async def test_cron_triggered_outreach_reuses_bound_plan_and_sets_canonical_trigger_fields(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock(
        return_value=SendResult(
            success=True,
            message_id="bridge-msg-cron",
            raw_response={
                "dispatch_group_id": "dispatch-cron",
                "messageId": "bridge-msg-cron",
            },
        )
    )

    await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."'
        )
    )

    existing_plan_id = load_whatsapp_outreach_run_records()[0]["plan"]["plan_id"]
    bind_whatsapp_outreach_plan_to_cron_job(
        plan_id=existing_plan_id,
        cron_job_id="cron-123",
    )

    from gateway.whatsapp_approved_outreach import execute_whatsapp_approved_outreach

    result = await execute_whatsapp_approved_outreach(
        {
            "workflow_binding_type": "whatsapp_outreach_plan",
            "workflow_binding_id": existing_plan_id,
            "trigger_source": "cron_job",
            "trigger_reference_id": "cron-123",
            "message_text": "Scheduled follow-up.",
            "report_delivery_target": "telegram:-100ops",
        },
        authorized=True,
        adapter=runner.adapters[Platform.WHATSAPP],
    )

    assert result["workflow_status"] == "ready"
    assert result["run"]["workflow_binding_type"] == "whatsapp_outreach_plan"
    assert result["run"]["workflow_binding_id"] == existing_plan_id
    assert result["run"]["trigger_source"] == "cron_job"
    assert result["run"]["trigger_reference_id"] == "cron-123"
    assert result["run"]["report_delivery_target"] == "telegram:-100ops"
    assert result["execution"]["resolved_target"]["destination_chat_id"] == (
        "15551230000@s.whatsapp.net"
    )

    runner.adapters[Platform.WHATSAPP].send.assert_awaited_with(
        "15551230000@s.whatsapp.net",
        "Scheduled follow-up.",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plan_status", "expected_reason"),
    [
        ("paused", "bound approved plan is paused"),
        ("cancelled", "bound approved plan is cancelled"),
        ("draft", "bound approved plan is unresolved"),
    ],
)
async def test_cron_triggered_outreach_fails_closed_for_unavailable_bound_plan(
    tmp_path, monkeypatch, plan_status, expected_reason
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock()

    await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."'
        )
    )
    runner.adapters[Platform.WHATSAPP].send.reset_mock()
    state = load_whatsapp_outreach_state()
    state["plans"][0]["plan_status"] = plan_status
    state["plans"][0]["linked_cron_job_id"] = "cron-123"
    from gateway.whatsapp_approved_outreach import _write_whatsapp_outreach_state

    _write_whatsapp_outreach_state(state)

    from gateway.whatsapp_approved_outreach import execute_whatsapp_approved_outreach

    result = await execute_whatsapp_approved_outreach(
        {
            "workflow_binding_type": "whatsapp_outreach_plan",
            "workflow_binding_id": state["plans"][0]["plan_id"],
            "trigger_source": "cron_job",
            "trigger_reference_id": "cron-123",
            "message_text": "Scheduled follow-up.",
        },
        authorized=True,
        adapter=runner.adapters[Platform.WHATSAPP],
    )

    runner.adapters[Platform.WHATSAPP].send.assert_not_awaited()
    assert result["workflow_status"] == "blocked"
    assert result["reason"] == expected_reason


@pytest.mark.asyncio
async def test_bound_plan_batch_run_creates_one_execution_per_active_target_and_partial_report(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _append_record(
        base_dir,
        record_id="record-1",
        text="Vendor A prior thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )
    append_whatsapp_record(
        {
            "record_kind": "conversation_record",
            "record_id": "record-2",
            "conversation_key": "whatsapp:dm:15551230001",
            "destination_key": "whatsapp:dm:15551230001",
            "destination_context_type": "direct_message",
            "destination_chat_id": "15551230001@s.whatsapp.net",
            "destination_target_id": "15551230001",
            "group_chat_id": None,
            "dm_counterparty_id": "15551230001",
            "direction": "inbound",
            "effective_event_at_utc": "2024-06-02T09:02:00Z",
            "record_sequence": 2,
            "participant_role": "external_party",
            "message_classification": "conversational_only",
            "command_authority_scope": "none",
            "sender_id": "15551230001",
            "sender_name": "Vendor B",
            "text": "Vendor B prior thread context.",
            "message_id": "msg-2",
            "media_types": [],
        },
        effective_event_at=datetime.fromisoformat("2024-06-02T09:02:00+00:00"),
        base_dir=base_dir,
    )

    runner = _make_runner(tmp_path)

    async def _send(chat_id, message):
        if chat_id == "15551230000@s.whatsapp.net":
            return SendResult(
                success=True,
                message_id="bridge-msg-a",
                raw_response={"dispatch_group_id": "dispatch-a"},
            )
        return SendResult(success=False, error="bridge down for vendor b")

    runner.adapters[Platform.WHATSAPP].send = AsyncMock(side_effect=_send)

    await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."'
        )
    )
    state = load_whatsapp_outreach_state()
    plan_id = state["plans"][0]["plan_id"]
    state["plan_targets"].append({
        "plan_target_id": "watarget-extra",
        "plan_id": plan_id,
        "target_status": "active",
        "conversation_key": "whatsapp:dm:15551230001",
        "destination_key": "whatsapp:dm:15551230001",
        "group_chat_id": None,
        "dm_counterparty_id": "15551230001",
        "target_objective_override": None,
        "max_outbound_messages_per_run": 1,
        "last_resolved_conversation_key": None,
        "last_observed_message_at_utc": None,
    })
    state["plans"][0]["plan_status"] = "approved"
    _write_whatsapp_outreach_state(state)
    bind_whatsapp_outreach_plan_to_cron_job(plan_id=plan_id, cron_job_id="cron-batch")

    from gateway.whatsapp_approved_outreach import execute_whatsapp_approved_outreach

    result = await execute_whatsapp_approved_outreach(
        {
            "workflow_binding_type": "whatsapp_outreach_plan",
            "workflow_binding_id": plan_id,
            "trigger_source": "cron_job",
            "trigger_reference_id": "cron-batch",
            "message_text": "Scheduled follow-up.",
            "report_delivery_target": "telegram:-100ops",
        },
        authorized=True,
        adapter=runner.adapters[Platform.WHATSAPP],
    )

    assert result["workflow_status"] == "ready"
    assert result["run"]["run_status"] == "completed_with_failures"
    assert result["run"]["target_count"] == 2
    assert result["run"]["completed_target_count"] == 2
    assert result["run"]["failed_target_count"] == 1
    assert len(result["run"]["target_executions"]) == 2

    statuses = {
        row["resolved_target"]["destination_chat_id"]: row["execution_status"]
        for row in result["run"]["target_executions"]
    }
    assert statuses["15551230000@s.whatsapp.net"] == "sent"
    assert statuses["15551230001@s.whatsapp.net"] == "send_failed"

    final_state = load_whatsapp_outreach_state()
    latest_run = final_state["runs"][-1]
    latest_report = final_state["reports"][-1]
    assert latest_run["run_status"] == "completed_with_failures"
    assert latest_run["target_count"] == 2
    assert (
        len([
            r
            for r in final_state["target_executions"]
            if r["run_id"] == latest_run["run_id"]
        ])
        == 2
    )
    assert latest_report["report_status"] == "partial"
    assert len(latest_report["target_rows"]) == 2
