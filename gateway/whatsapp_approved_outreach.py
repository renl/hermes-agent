from __future__ import annotations

import json
import os
import shlex
import threading
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.skill_commands import _build_skill_message, _load_skill_payload
from hermes_constants import get_hermes_home

from gateway.whatsapp_message_store import (
    append_whatsapp_record,
    build_whatsapp_destination_fields,
    query_whatsapp_records_any_time,
    utc_isoformat,
    utc_now,
)
from gateway.whatsapp_transcript_summary import ResolveWhatsAppExactTarget

WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS = (
    "conversation_key",
    "destination_key",
    "group_chat_id",
    "dm_counterparty_id",
)
WHATSAPP_APPROVED_OUTREACH_ALLOWED_FIELDS = (
    *WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS,
    "approved_destination_chat_id",
    "operator_objective",
    "message_text",
    "report_delivery_target",
    "workflow_binding_type",
    "workflow_binding_id",
)
_OUTREACH_PREFIXES = (
    "whatsapp outreach",
    "whatsapp-outreach",
)
_LOCAL_GOVERNANCE_PREFIXES = (
    "whatsapp start",
    "whatsapp revise",
    "whatsapp conversation start",
    "whatsapp conversation revise",
)
WHATSAPP_OUTREACH_STATE_SCHEMA_VERSION = 1
WHATSAPP_OUTREACH_PLAN_WORKFLOW_BINDING_TYPE = "whatsapp_outreach_plan"
WHATSAPP_DEFAULT_COMMUNICATION_SKILL_NAME = "whatsapp-communications"
WHATSAPP_ALLOWED_OPERATOR_INGRESS_SURFACES = (
    "cli_chat",
    "tui_chat",
    "dashboard_chat",
)
WHATSAPP_ALLOWED_INTERACTION_LANES = (
    "external_conversation",
    "operator_governance",
    "admin_debug_report",
)
WHATSAPP_CONVERSATION_CHANNEL_MODE = "conversational_primary"
WHATSAPP_OPERATOR_GOVERNANCE_POLICY = "owner_approved_conversation"
WHATSAPP_ALLOWED_COMMUNICATION_STAGES = (
    "opening_outreach",
    "follow_up",
    "clarification",
    "negotiation",
    "closeout",
)
WHATSAPP_ALLOWED_DRAFT_OUTCOMES = (
    "send_message",
    "no_send_required",
    "handoff_required",
)
_JSON_PRIMITIVE_TYPES = (str, int, float, bool, type(None))

_OUTREACH_STATE_LOCK = threading.Lock()


def build_local_chat_whatsapp_outreach_requests(
    text: str | None,
    *,
    session_id: str | None,
    operator_ingress_surface: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed_request = parse_whatsapp_approved_outreach_instruction(text)
    normalized_instruction = normalize_whatsapp_approved_outreach_request({
        **parsed_request,
        "trigger_source": "owner_instruction",
        "trigger_reference_id": session_id,
        "operator_ingress_surface": operator_ingress_surface,
        "interaction_lane": "operator_governance",
        "conversation_channel_mode": WHATSAPP_CONVERSATION_CHANNEL_MODE,
        "operator_governance_policy": WHATSAPP_OPERATOR_GOVERNANCE_POLICY,
    })
    operator_instruction = dict(normalized_instruction)
    return operator_instruction, dict(operator_instruction)


def build_cli_chat_whatsapp_outreach_requests(
    text: str | None,
    *,
    session_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return build_local_chat_whatsapp_outreach_requests(
        text,
        session_id=session_id,
        operator_ingress_surface="cli_chat",
    )


def _matches_instruction_prefix(text: str | None, prefixes: tuple[str, ...]) -> bool:
    normalized = str(text or "").strip().lower()
    return any(
        normalized == prefix or normalized.startswith(f"{prefix} ")
        for prefix in prefixes
    )


def is_whatsapp_local_governance_instruction(text: str | None) -> bool:
    return _matches_instruction_prefix(text, _LOCAL_GOVERNANCE_PREFIXES)


def is_whatsapp_legacy_outreach_instruction(text: str | None) -> bool:
    return _matches_instruction_prefix(text, _OUTREACH_PREFIXES)


def is_whatsapp_approved_outreach_instruction(text: str | None) -> bool:
    return is_whatsapp_local_governance_instruction(
        text
    ) or is_whatsapp_legacy_outreach_instruction(text)


def parse_whatsapp_approved_outreach_instruction(text: str | None) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    lowered = raw_text.lower()
    prefix = next(
        (
            candidate
            for candidate in (*_LOCAL_GOVERNANCE_PREFIXES, *_OUTREACH_PREFIXES)
            if lowered == candidate or lowered.startswith(f"{candidate} ")
        ),
        None,
    )
    if prefix is None:
        raise ValueError("instruction does not match WhatsApp approved-outreach syntax")

    raw_args = raw_text[len(prefix) :].strip()
    if not raw_args:
        return {}

    parsed: dict[str, Any] = {}
    for token in shlex.split(raw_args):
        if "=" not in token:
            raise ValueError("instruction arguments must use field=value syntax")
        field, value = token.split("=", 1)
        normalized_field = field.strip()
        if normalized_field not in WHATSAPP_APPROVED_OUTREACH_ALLOWED_FIELDS:
            raise ValueError(f"unsupported field: {normalized_field}")
        parsed[normalized_field] = value.strip()
    return parsed


def normalize_whatsapp_approved_outreach_request(
    request: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_request = request or {}
    workflow_binding_type = str(raw_request.get("workflow_binding_type") or "").strip()
    workflow_binding_id = str(raw_request.get("workflow_binding_id") or "").strip()
    normalized = {
        field: (str(raw_request.get(field) or "").strip() or None)
        for field in WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS
    }
    selected_fields = [field for field, value in normalized.items() if value]
    approved_destination_chat_id = (
        str(raw_request.get("approved_destination_chat_id") or "").strip() or None
    )
    if not workflow_binding_type:
        if approved_destination_chat_id and selected_fields:
            raise ValueError("approved outreach requires exactly one target basis")
        if not approved_destination_chat_id and len(selected_fields) != 1:
            raise ValueError("approved outreach requires exactly one exact selector")

    operator_objective = str(raw_request.get("operator_objective") or "").strip()
    if bool(workflow_binding_type) != bool(workflow_binding_id):
        raise ValueError(
            "workflow_binding_type and workflow_binding_id must be provided together"
        )
    if (
        workflow_binding_type
        and workflow_binding_type != WHATSAPP_OUTREACH_PLAN_WORKFLOW_BINDING_TYPE
    ):
        raise ValueError(f"unsupported workflow_binding_type: {workflow_binding_type}")

    if not operator_objective and not workflow_binding_type:
        raise ValueError("operator_objective is required")

    message_text = str(raw_request.get("message_text") or "").strip() or None
    report_delivery_target = (
        str(raw_request.get("report_delivery_target") or "").strip() or None
    )
    trigger_source = str(raw_request.get("trigger_source") or "").strip()
    trigger_reference_id = (
        str(raw_request.get("trigger_reference_id") or "").strip() or None
    )
    operator_ingress_surface = (
        str(raw_request.get("operator_ingress_surface") or "").strip() or None
    )
    interaction_lane = str(raw_request.get("interaction_lane") or "").strip()
    if not interaction_lane:
        interaction_lane = "operator_governance"
    if interaction_lane not in WHATSAPP_ALLOWED_INTERACTION_LANES:
        raise ValueError(f"unsupported interaction_lane: {interaction_lane}")
    conversation_channel_mode = (
        str(raw_request.get("conversation_channel_mode") or "").strip()
        or WHATSAPP_CONVERSATION_CHANNEL_MODE
    )
    if conversation_channel_mode != WHATSAPP_CONVERSATION_CHANNEL_MODE:
        raise ValueError("conversation_channel_mode must be conversational_primary")
    operator_governance_policy = (
        str(raw_request.get("operator_governance_policy") or "").strip()
        or WHATSAPP_OPERATOR_GOVERNANCE_POLICY
    )
    if operator_governance_policy != WHATSAPP_OPERATOR_GOVERNANCE_POLICY:
        raise ValueError(
            "operator_governance_policy must be owner_approved_conversation"
        )
    if (
        operator_ingress_surface is not None
        and operator_ingress_surface not in WHATSAPP_ALLOWED_OPERATOR_INGRESS_SURFACES
    ):
        raise ValueError(
            f"unsupported operator_ingress_surface: {operator_ingress_surface}"
        )
    if not trigger_source:
        trigger_source = "owner_instruction"
    return {
        **normalized,
        "approved_destination_chat_id": approved_destination_chat_id,
        "operator_objective": operator_objective or None,
        "message_text": message_text,
        "report_delivery_target": report_delivery_target,
        "trigger_source": trigger_source,
        "trigger_reference_id": trigger_reference_id,
        "operator_ingress_surface": operator_ingress_surface,
        "interaction_lane": interaction_lane,
        "conversation_channel_mode": conversation_channel_mode,
        "operator_governance_policy": operator_governance_policy,
        "workflow_binding_type": workflow_binding_type or None,
        "workflow_binding_id": workflow_binding_id or None,
    }


def _ensure_plan_contract_fields(plan_row: dict[str, Any]) -> dict[str, Any]:
    plan_row["interaction_lane"] = "operator_governance"
    plan_row["conversation_channel_mode"] = WHATSAPP_CONVERSATION_CHANNEL_MODE
    plan_row["operator_governance_policy"] = WHATSAPP_OPERATOR_GOVERNANCE_POLICY
    plan_row.setdefault("used_legacy_whatsapp_command_ingress", False)
    return plan_row


def _ensure_run_contract_fields(
    run: dict[str, Any], *, plan: dict[str, Any]
) -> dict[str, Any]:
    run["interaction_lane"] = "operator_governance"
    run["conversation_channel_mode"] = plan.get(
        "conversation_channel_mode", WHATSAPP_CONVERSATION_CHANNEL_MODE
    )
    run["operator_governance_policy"] = plan.get(
        "operator_governance_policy", WHATSAPP_OPERATOR_GOVERNANCE_POLICY
    )
    return run


def _ensure_execution_contract_fields(execution: dict[str, Any]) -> dict[str, Any]:
    execution["interaction_lane"] = "external_conversation"
    return execution


def _ensure_report_row_contract_fields(row: dict[str, Any]) -> dict[str, Any]:
    row["interaction_lane"] = "external_conversation"
    return row


def _ensure_report_contract_fields(report: dict[str, Any]) -> dict[str, Any]:
    report["interaction_lane"] = "admin_debug_report"
    return report


def _normalize_existing_outreach_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if (
        normalized.get("report_status") is not None
        or normalized.get("target_rows") is not None
    ):
        return _ensure_report_contract_fields(normalized)
    if normalized.get("run_status") is not None:
        plan_like = {
            "conversation_channel_mode": normalized.get(
                "conversation_channel_mode", WHATSAPP_CONVERSATION_CHANNEL_MODE
            ),
            "operator_governance_policy": normalized.get(
                "operator_governance_policy", WHATSAPP_OPERATOR_GOVERNANCE_POLICY
            ),
        }
        return _ensure_run_contract_fields(normalized, plan=plan_like)
    if normalized.get("execution_status") is not None:
        return _ensure_execution_contract_fields(normalized)
    if normalized.get("plan_status") is not None:
        return _ensure_plan_contract_fields(normalized)
    if normalized.get("observable_status") is not None:
        return _ensure_report_row_contract_fields(normalized)
    return normalized


def _outreach_store_path(*, base_dir: Path | None = None) -> Path:
    hermes_home = base_dir or get_hermes_home()
    return hermes_home / "gateway" / "whatsapp-approved-outreach-state.json"


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, _JSON_PRIMITIVE_TYPES):
        return value
    if isinstance(value, (date, datetime, time)):
        try:
            return value.isoformat()
        except Exception:
            return None
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return None


def _json_safe_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text or None
    if isinstance(value, (date, datetime, time)):
        try:
            text = value.isoformat().strip()
        except Exception:
            return None
        return text or None
    return None


def _safe_instance_attr(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict):
        return value_dict.get(field_name)
    return None


def _resolve_send_callable(adapter: Any) -> Any:
    adapter_dict = getattr(adapter, "__dict__", {})
    send_defined_on_type = any("send" in cls.__dict__ for cls in type(adapter).__mro__)
    if "send" in adapter_dict or send_defined_on_type:
        send_callable = getattr(adapter, "send", None)
        if callable(send_callable):
            return send_callable
    if callable(adapter):
        return adapter
    return None


def _default_outreach_state() -> dict[str, Any]:
    return {
        "schema_version": WHATSAPP_OUTREACH_STATE_SCHEMA_VERSION,
        "plans": [],
        "plan_targets": [],
        "runs": [],
        "target_executions": [],
        "reports": [],
    }


def _normalized_outreach_state(raw_state: dict[str, Any] | None) -> dict[str, Any]:
    state = _default_outreach_state()
    if not isinstance(raw_state, dict):
        return state

    schema_version = raw_state.get("schema_version")
    if isinstance(schema_version, int) and schema_version > 0:
        state["schema_version"] = schema_version

    for field in ("plans", "plan_targets", "runs", "target_executions", "reports"):
        value = raw_state.get(field)
        if isinstance(value, list):
            state[field] = [
                _normalize_existing_outreach_row(item)
                for item in value
                if isinstance(item, dict)
            ]
    return state


def load_whatsapp_outreach_state(*, base_dir: Path | None = None) -> dict[str, Any]:
    path = _outreach_store_path(base_dir=base_dir)
    if not path.exists():
        return _default_outreach_state()

    with path.open("r", encoding="utf-8") as handle:
        return _normalized_outreach_state(json.load(handle))


def _write_whatsapp_outreach_state(
    state: dict[str, Any], *, base_dir: Path | None = None
) -> Path:
    path = _outreach_store_path(base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _json_safe_value(state),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _find_first(
    rows: list[dict[str, Any]], field_name: str, expected: str | None
) -> dict[str, Any] | None:
    for row in rows:
        if row.get(field_name) == expected:
            return row
    return None


def _exact_selector_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: (str(row.get(field) or "").strip() or None)
        for field in WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS
    }


def _resolved_target_from_execution(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_key": execution.get("resolved_conversation_key"),
        "destination_key": execution.get("resolved_destination_key"),
        "group_chat_id": execution.get("group_chat_id"),
        "dm_counterparty_id": execution.get("dm_counterparty_id"),
        "destination_context_type": execution.get("destination_context_type"),
        "destination_chat_id": execution.get("resolved_destination_chat_id"),
        "destination_target_id": execution.get("destination_target_id"),
    }


def _resolve_target_from_preserved_history(
    selector: dict[str, Any],
) -> dict[str, Any]:
    continuity_rows = query_whatsapp_records_any_time(
        conversation_key=selector.get("conversation_key"),
        destination_key=selector.get("destination_key"),
        destination_context_type=selector.get("destination_context_type"),
        group_chat_id=selector.get("group_chat_id"),
        dm_counterparty_id=selector.get("dm_counterparty_id"),
    )
    if not continuity_rows:
        return {
            "resolution_status": "target_not_found",
            "resolved_target": None,
        }

    exact_targets: dict[
        tuple[str | None, str | None, str | None, str | None],
        dict[str, Any],
    ] = {}
    for row in continuity_rows:
        target_key = tuple(
            str(row.get(field) or "").strip() or None
            for field in WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS
        )
        exact_targets.setdefault(
            target_key,
            {
                "conversation_key": target_key[0],
                "destination_key": target_key[1],
                "group_chat_id": target_key[2],
                "dm_counterparty_id": target_key[3],
                "destination_context_type": (
                    str(row.get("destination_context_type") or "").strip() or None
                ),
                "destination_chat_id": (
                    str(row.get("destination_chat_id") or "").strip() or None
                ),
                "destination_target_id": (
                    str(row.get("destination_target_id") or "").strip() or None
                ),
            },
        )

    if len(exact_targets) != 1:
        return {
            "resolution_status": "target_not_found",
            "resolved_target": None,
        }

    resolved_target = next(iter(exact_targets.values()))
    if not resolved_target.get("destination_chat_id"):
        return {
            "resolution_status": "target_not_found",
            "resolved_target": None,
        }

    return {
        "resolution_status": "resolved",
        "resolved_target": resolved_target,
    }


def _enriched_plan(
    state: dict[str, Any], plan: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not isinstance(plan, dict):
        return None
    enriched = _ensure_plan_contract_fields(dict(plan))
    enriched["approved_targets"] = [
        dict(target)
        for target in state["plan_targets"]
        if target.get("plan_id") == plan.get("plan_id")
    ]
    return enriched


def _enriched_execution(execution: dict[str, Any]) -> dict[str, Any]:
    enriched = _ensure_execution_contract_fields(dict(execution))
    enriched["resolved_target"] = _resolved_target_from_execution(execution)
    return enriched


def _enriched_run(
    state: dict[str, Any], run: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not isinstance(run, dict):
        return None
    plan = _find_first(state["plans"], "plan_id", run.get("plan_id")) or {}
    enriched = _ensure_run_contract_fields(dict(run), plan=plan)
    enriched["target_executions"] = [
        _enriched_execution(execution)
        for execution in state["target_executions"]
        if execution.get("run_id") == run.get("run_id")
    ]
    return enriched


def _matching_plan_and_target(
    state: dict[str, Any],
    *,
    operator_objective: str,
    approved_by_principal: str,
    selector: dict[str, Any],
    approved_destination_chat_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    selected_fields = [field for field, value in selector.items() if value]
    for plan_target in reversed(state["plan_targets"]):
        if plan_target.get("target_status") != "active":
            continue
        if approved_destination_chat_id is not None:
            if (
                plan_target.get("approved_destination_chat_id")
                != approved_destination_chat_id
            ):
                continue
        elif plan_target.get("approved_destination_chat_id"):
            continue
        if any(
            plan_target.get(field) != selector.get(field) for field in selected_fields
        ):
            continue
        plan = _find_first(state["plans"], "plan_id", plan_target.get("plan_id"))
        if plan is None:
            continue
        if plan.get("approved_by_principal") != approved_by_principal:
            continue
        if plan.get("operator_objective") != operator_objective:
            continue
        if plan.get("trigger_mode") != "instruction_only":
            continue
        if plan.get("plan_status") not in {"approved", "active"}:
            continue
        return plan, plan_target
    return None, None


def _active_targets_for_plan(
    state: dict[str, Any], plan_id: str
) -> list[dict[str, Any]]:
    return [
        dict(target)
        for target in state["plan_targets"]
        if target.get("plan_id") == plan_id and target.get("target_status") == "active"
    ]


def _determine_plan_trigger_mode(
    active_target_count: int, *, has_cron_binding: bool
) -> str:
    if has_cron_binding:
        return "instruction_and_cron"
    return "instruction_only" if active_target_count <= 1 else "instruction_and_cron"


def _status_basis_for_observable_status(observable_status: str) -> str:
    if observable_status in {
        "awaiting_reply",
        "follow_up_recommended",
        "unresolved",
        "send_failed",
    }:
        return "bounded_model_synthesis"
    return "direct_conversation_evidence"


def _observable_status_from_history_and_execution(
    history_rows: list[dict[str, Any]], execution_status: str
) -> str:
    if execution_status == "handoff_required":
        return "handoff_required"
    if execution_status == "send_failed":
        return "send_failed"
    if execution_status == "sent":
        return "awaiting_reply"
    latest_external_text = ""
    for row in reversed(history_rows):
        if row.get("participant_role") == "external_party":
            latest_external_text = str(row.get("text") or "").lower()
            break
    if "quote" in latest_external_text:
        return "quote_received"
    if any(
        token in latest_external_text
        for token in ("negot", "counter", "discount", "price")
    ):
        return "negotiating"
    if latest_external_text:
        return "reply_received"
    if execution_status == "no_send_required":
        return "follow_up_recommended"
    return "unresolved"


def _execution_summary_line(execution: dict[str, Any]) -> str:
    resolved_target = execution.get("resolved_target") or {}
    destination_label = (
        resolved_target.get("destination_target_id")
        or resolved_target.get("destination_key")
        or resolved_target.get("conversation_key")
        or execution.get("plan_target_id")
    )
    status = execution.get("execution_status") or "unknown"
    error = execution.get("last_error")
    if error:
        return f"- {destination_label}: {status} ({error})"
    return f"- {destination_label}: {status}"


def _founder_visible_ingress_summary(
    *,
    operator_ingress_surface: str | None,
    interaction_lane: str | None,
    used_legacy_whatsapp_command_ingress: bool,
) -> str:
    surface = operator_ingress_surface or "unknown"
    lane = interaction_lane or "unknown"
    if used_legacy_whatsapp_command_ingress:
        return (
            "Founder-visible ingress is now expected on Hermes-local governance surfaces; "
            "this run used legacy WhatsApp command-shaped governance as a supporting path only. "
            f"Recorded governance lane={lane}, operator_ingress_surface={surface}."
        )
    return (
        "Founder-visible ingress ran through Hermes-local governance and then handed off to the "
        f"normal external conversation send lane. Recorded governance lane={lane}, "
        f"operator_ingress_surface={surface}."
    )


def _blocked_result(*, founder_summary: str, reason: str) -> dict[str, Any]:
    return _result(
        workflow_status="blocked",
        founder_summary=founder_summary,
        reason=reason,
    )


def _validate_bound_plan(
    state: dict[str, Any],
    *,
    workflow_binding_id: str,
    trigger_reference_id: str | None,
    require_linked_cron_job_match: bool = True,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None, str | None, str | None]:
    plan_row = _find_first(state["plans"], "plan_id", workflow_binding_id)
    if plan_row is None:
        return (
            None,
            None,
            "WhatsApp approved outreach stayed blocked because the bound approved plan could not be found.",
            "bound approved plan not found",
        )

    plan_status = str(plan_row.get("plan_status") or "").strip().lower()
    if plan_status == "paused":
        return (
            None,
            None,
            "WhatsApp approved outreach stayed blocked because the bound approved plan is paused.",
            "bound approved plan is paused",
        )
    if plan_status == "cancelled":
        return (
            None,
            None,
            "WhatsApp approved outreach stayed blocked because the bound approved plan is cancelled.",
            "bound approved plan is cancelled",
        )
    if plan_status not in {"approved", "active"}:
        return (
            None,
            None,
            "WhatsApp approved outreach stayed blocked because the bound approved plan is unresolved.",
            "bound approved plan is unresolved",
        )

    linked_cron_job_id = str(plan_row.get("linked_cron_job_id") or "").strip() or None
    if require_linked_cron_job_match and (
        not linked_cron_job_id or linked_cron_job_id != trigger_reference_id
    ):
        return (
            None,
            None,
            "WhatsApp approved outreach stayed blocked because the bound approved plan is unresolved for this cron job.",
            "bound approved plan is unresolved for this cron job",
        )

    active_targets = _active_targets_for_plan(state, workflow_binding_id)
    if not active_targets:
        return (
            None,
            None,
            "WhatsApp approved outreach stayed blocked because the bound approved plan has no active approved targets.",
            "bound approved plan has no active approved targets",
        )

    for target in active_targets:
        if target.get("max_outbound_messages_per_run") != 1:
            return (
                None,
                None,
                "WhatsApp approved outreach stayed blocked because the bound approved plan exceeds the single-target v1 send limit for at least one approved target.",
                "bound approved plan exceeds the single-target v1 send limit",
            )

    return plan_row, active_targets, None, None


def _resolve_bound_plan_request(
    normalized_request: dict[str, Any], *, state: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    workflow_binding_type = normalized_request.get("workflow_binding_type")
    if workflow_binding_type != WHATSAPP_OUTREACH_PLAN_WORKFLOW_BINDING_TYPE:
        return normalized_request, None, None

    plan_row, active_targets, founder_summary, reason = _validate_bound_plan(
        state,
        workflow_binding_id=str(normalized_request["workflow_binding_id"]),
        trigger_reference_id=normalized_request.get("trigger_reference_id"),
    )
    if plan_row is None or active_targets is None:
        return (
            None,
            None,
            _blocked_result(
                founder_summary=str(founder_summary),
                reason=str(reason),
            ),
        )
    operator_objective = str(plan_row.get("operator_objective") or "").strip()
    if not operator_objective:
        return (
            None,
            None,
            _blocked_result(
                founder_summary=(
                    "WhatsApp approved outreach stayed blocked because the bound approved plan has no resolved operator objective."
                ),
                reason="bound approved plan has no resolved operator objective",
            ),
        )

    return (
        {**normalized_request, "operator_objective": operator_objective},
        plan_row,
        None,
    )


def _observable_status_for_execution_status(execution_status: str) -> str:
    if execution_status == "sent":
        return "awaiting_reply"
    if execution_status == "send_failed":
        return "send_failed"
    return "unresolved"


def _status_basis_for_execution_status(execution_status: str) -> str:
    if execution_status == "sent":
        return "bounded_model_synthesis"
    return "direct_conversation_evidence"


def _normalized_communication_skill_name(value: Any) -> str:
    return str(value or "").strip() or WHATSAPP_DEFAULT_COMMUNICATION_SKILL_NAME


def _select_communication_stage(history_rows: list[dict[str, Any]]) -> str:
    if any(row.get("participant_role") == "agent" for row in history_rows):
        return "follow_up"
    return "opening_outreach"


def _cold_start_failure_reason(
    *,
    approved_destination_chat_id: str | None,
    destination_fields: dict[str, Any] | None,
) -> str | None:
    if not approved_destination_chat_id:
        return (
            "approved_destination_chat_id is required for cold-start DM first contact"
        )
    if not isinstance(destination_fields, dict):
        return "approved_destination_chat_id could not be normalized"
    if destination_fields.get("destination_context_type") != "direct_message":
        return (
            "approved_destination_chat_id must represent one direct-message destination"
        )
    if not destination_fields.get("destination_chat_id"):
        return (
            "approved_destination_chat_id must resolve to one exact destination_chat_id"
        )
    if not destination_fields.get("dm_counterparty_id"):
        return (
            "approved_destination_chat_id must resolve to one exact dm_counterparty_id"
        )
    return None


def _persist_cold_start_outbound_record(
    *,
    approved_destination_chat_id_snapshot: str,
    outbound_message_text: str,
    message_id: str | None,
    effective_event_at_utc: str,
) -> None:
    effective_event_at = datetime.fromisoformat(
        effective_event_at_utc.replace("Z", "+00:00")
    )
    destination_fields = build_whatsapp_destination_fields(
        approved_destination_chat_id_snapshot
    )
    record_id = f"wa-cold-start-{uuid4()}"
    append_whatsapp_record(
        {
            "record_kind": "conversation_record",
            "record_id": record_id,
            "conversation_key": destination_fields.get("destination_key"),
            **destination_fields,
            "direction": "outbound",
            "effective_event_at_utc": effective_event_at_utc,
            "record_sequence": 0,
            "participant_role": "agent",
            "message_classification": "conversational_only",
            "command_authority_scope": "none",
            "sender_id": "agent",
            "sender_name": "Hermes",
            "text": outbound_message_text,
            "message_id": message_id,
            "media_types": [],
        },
        effective_event_at=effective_event_at,
    )


def _parse_draft_response_payload(raw_response: str) -> dict[str, Any]:
    text = str(raw_response or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("communication draft response was not valid JSON")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("communication draft response must be a JSON object")
    return payload


def _normalize_whatsapp_communication_draft(
    raw_draft: dict[str, Any],
) -> dict[str, Any]:
    draft_outcome = str(raw_draft.get("draft_outcome") or "").strip()
    if draft_outcome not in WHATSAPP_ALLOWED_DRAFT_OUTCOMES:
        raise ValueError(f"unsupported draft_outcome: {draft_outcome or '<empty>'}")

    outbound_message_text = (
        str(raw_draft.get("outbound_message_text") or "").strip() or None
    )
    handoff_reason = str(raw_draft.get("handoff_reason") or "").strip() or None

    if draft_outcome == "send_message":
        if not outbound_message_text:
            raise ValueError("send_message draft requires outbound_message_text")
        if handoff_reason is not None:
            raise ValueError("send_message draft must not include handoff_reason")
    elif draft_outcome == "handoff_required":
        if not handoff_reason:
            raise ValueError("handoff_required draft requires handoff_reason")
        if outbound_message_text is not None:
            raise ValueError(
                "handoff_required draft must not include outbound_message_text"
            )
    else:
        if outbound_message_text is not None:
            raise ValueError(
                "no_send_required draft must not include outbound_message_text"
            )
        if handoff_reason is not None:
            raise ValueError("no_send_required draft must not include handoff_reason")

    return {
        "draft_outcome": draft_outcome,
        "outbound_message_text": outbound_message_text,
        "handoff_reason": handoff_reason,
    }


def _build_whatsapp_communication_draft_request(
    *,
    communication_skill_name: str,
    operator_objective: str,
    resolved_target: dict[str, Any],
    history_rows: list[dict[str, Any]],
    message_text_override: str | None,
    trigger_source: str,
) -> dict[str, Any]:
    communication_stage = _select_communication_stage(history_rows)
    if communication_stage not in WHATSAPP_ALLOWED_COMMUNICATION_STAGES:
        raise ValueError(f"unsupported communication_stage: {communication_stage}")
    return {
        "communication_skill_name": communication_skill_name,
        "communication_stage": communication_stage,
        "operator_objective": operator_objective,
        "resolved_target": _json_safe_value(resolved_target),
        "history_rows": _json_safe_value(history_rows),
        "message_text_override": message_text_override,
        "trigger_source": trigger_source,
    }


def _draft_whatsapp_message_via_skill(
    *,
    communication_skill_name: str,
    operator_objective: str,
    resolved_target: dict[str, Any],
    history_rows: list[dict[str, Any]],
    message_text_override: str | None,
    trigger_source: str,
    task_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = _load_skill_payload(communication_skill_name, task_id=task_id)
    if not loaded:
        raise ValueError(
            f"communication skill '{communication_skill_name}' could not be loaded"
        )

    loaded_skill, skill_dir, skill_name = loaded
    skill_message = _build_skill_message(
        loaded_skill,
        skill_dir,
        (
            f'[IMPORTANT: The "{skill_name}" skill is loaded for this exact '
            "WhatsApp approved-outreach execution. Follow it for this one "
            "bounded communication decision.]"
        ),
        runtime_note=(
            "Decide one bounded WhatsApp communication outcome for one already "
            "resolved target. Do not resolve targets, widen scope, or call the "
            "WhatsApp bridge."
        ),
        session_id=task_id,
    )
    draft_request = _build_whatsapp_communication_draft_request(
        communication_skill_name=communication_skill_name,
        operator_objective=operator_objective,
        resolved_target=resolved_target,
        history_rows=history_rows,
        message_text_override=message_text_override,
        trigger_source=trigger_source,
    )

    from run_agent import AIAgent

    agent = AIAgent(
        quiet_mode=True,
        skip_memory=True,
        platform="whatsapp",
        session_id=task_id,
    )
    result = agent.run_conversation(
        (
            "Return one JSON object only for this WhatsApp communication draft.\n"
            "Allowed draft_outcome values: send_message, no_send_required, "
            "handoff_required.\n"
            "Include outbound_message_text only for send_message.\n"
            "Include handoff_reason only for handoff_required.\n"
            "Do not include markdown fences or extra prose.\n\n"
            "WhatsAppCommunicationDraftRequest:\n"
            f"{json.dumps(draft_request, ensure_ascii=False, indent=2)}"
        ),
        system_message=(
            "You are producing a bounded WhatsApp communication draft for Hermes. "
            "Follow the loaded skill. Return valid JSON only."
        ),
        conversation_history=[{"role": "user", "content": skill_message}],
        task_id=task_id,
    )
    final_response = str((result or {}).get("final_response") or "")
    return draft_request, _normalize_whatsapp_communication_draft(
        _parse_draft_response_payload(final_response)
    )


def _persist_whatsapp_outreach_run_record(record: dict[str, Any]) -> Path:
    with _OUTREACH_STATE_LOCK:
        state = load_whatsapp_outreach_state()
        plan = record.get("plan") if isinstance(record, dict) else None
        run = record.get("run") if isinstance(record, dict) else None
        report = record.get("report") if isinstance(record, dict) else None
        if isinstance(plan, dict):
            plan_targets = plan.get("approved_targets") or []
            state["plans"] = [
                row
                for row in state["plans"]
                if row.get("plan_id") != plan.get("plan_id")
            ]
            state["plans"].append({
                key: value for key, value in plan.items() if key != "approved_targets"
            })
            _ensure_plan_contract_fields(state["plans"][-1])
            if isinstance(plan_targets, list):
                target_ids = {
                    target.get("plan_target_id")
                    for target in plan_targets
                    if isinstance(target, dict)
                }
                state["plan_targets"] = [
                    row
                    for row in state["plan_targets"]
                    if row.get("plan_target_id") not in target_ids
                ]
                state["plan_targets"].extend(
                    dict(target) for target in plan_targets if isinstance(target, dict)
                )
        if isinstance(run, dict):
            target_executions = run.get("target_executions") or []
            state["runs"] = [
                row for row in state["runs"] if row.get("run_id") != run.get("run_id")
            ]
            state["runs"].append({
                key: value for key, value in run.items() if key != "target_executions"
            })
            _ensure_run_contract_fields(state["runs"][-1], plan=plan or {})
            if isinstance(target_executions, list):
                execution_ids = {
                    execution.get("target_execution_id")
                    for execution in target_executions
                    if isinstance(execution, dict)
                }
                state["target_executions"] = [
                    row
                    for row in state["target_executions"]
                    if row.get("target_execution_id") not in execution_ids
                ]
                for execution in target_executions:
                    if not isinstance(execution, dict):
                        continue
                    execution_row = dict(execution)
                    resolved_target = execution_row.pop("resolved_target", None)
                    if isinstance(resolved_target, dict):
                        execution_row.setdefault(
                            "resolved_conversation_key",
                            resolved_target.get("conversation_key"),
                        )
                        execution_row.setdefault(
                            "resolved_destination_key",
                            resolved_target.get("destination_key"),
                        )
                        execution_row.setdefault(
                            "resolved_destination_chat_id",
                            resolved_target.get("destination_chat_id"),
                        )
                        execution_row.setdefault(
                            "group_chat_id", resolved_target.get("group_chat_id")
                        )
                        execution_row.setdefault(
                            "dm_counterparty_id",
                            resolved_target.get("dm_counterparty_id"),
                        )
                        execution_row.setdefault(
                            "destination_context_type",
                            resolved_target.get("destination_context_type"),
                        )
                        execution_row.setdefault(
                            "destination_target_id",
                            resolved_target.get("destination_target_id"),
                        )
                    _ensure_execution_contract_fields(execution_row)
                    state["target_executions"].append(execution_row)
        if isinstance(report, dict):
            state["reports"] = [
                row
                for row in state["reports"]
                if row.get("report_id") != report.get("report_id")
            ]
            state["reports"].append(_ensure_report_contract_fields(dict(report)))
        return _write_whatsapp_outreach_state(state)


def append_whatsapp_outreach_run_record(record: dict[str, Any]) -> Path:
    return _persist_whatsapp_outreach_run_record(record)


def load_whatsapp_outreach_run_records(
    *, base_dir: Path | None = None
) -> list[dict[str, Any]]:
    state = load_whatsapp_outreach_state(base_dir=base_dir)
    results: list[dict[str, Any]] = []
    for run in state["runs"]:
        plan = _find_first(state["plans"], "plan_id", run.get("plan_id"))
        report = _find_first(state["reports"], "report_id", run.get("report_id"))
        results.append({
            "recorded_at_utc": run.get("run_completed_at_utc")
            or run.get("run_started_at_utc"),
            "plan": _enriched_plan(state, plan),
            "run": _enriched_run(state, run),
            "report": dict(report) if isinstance(report, dict) else None,
        })
    return results


def _result(
    *,
    workflow_status: str,
    plan: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    founder_summary: str = "",
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "workflow_status": workflow_status,
        "plan": plan,
        "run": run,
        "execution": execution,
        "founder_summary": founder_summary,
        "reason": reason,
    }


def bind_whatsapp_outreach_plan_to_cron_job(
    *, plan_id: str, cron_job_id: str
) -> dict[str, Any]:
    with _OUTREACH_STATE_LOCK:
        state = load_whatsapp_outreach_state()
        plan_row, active_targets, _founder_summary, reason = _validate_bound_plan(
            state,
            workflow_binding_id=plan_id,
            trigger_reference_id=cron_job_id,
            require_linked_cron_job_match=False,
        )
        if plan_row is None or active_targets is None:
            raise ValueError(str(reason))

        plan_row["linked_cron_job_id"] = cron_job_id
        plan_row["trigger_mode"] = _determine_plan_trigger_mode(
            len(active_targets), has_cron_binding=True
        )
        state["plans"] = [
            row for row in state["plans"] if row.get("plan_id") != plan_row["plan_id"]
        ]
        state["plans"].append(dict(plan_row))
        _write_whatsapp_outreach_state(state)
        return dict(plan_row)


def unbind_whatsapp_outreach_plan_from_cron_job(*, cron_job_id: str) -> None:
    with _OUTREACH_STATE_LOCK:
        state = load_whatsapp_outreach_state()
        changed = False
        for plan_row in state["plans"]:
            if plan_row.get("linked_cron_job_id") == cron_job_id:
                plan_row["linked_cron_job_id"] = None
                active_targets = _active_targets_for_plan(
                    state, str(plan_row.get("plan_id") or "")
                )
                plan_row["trigger_mode"] = _determine_plan_trigger_mode(
                    len(active_targets), has_cron_binding=False
                )
                changed = True
        if changed:
            _write_whatsapp_outreach_state(state)


def sync_whatsapp_outreach_plan_cron_bindings(jobs: list[dict[str, Any]]) -> None:
    bound_pairs = {
        (
            str(job.get("workflow_binding_id") or "").strip(),
            str(job.get("id") or job.get("job_id") or "").strip(),
        )
        for job in jobs
        if str(job.get("workflow_binding_type") or "").strip()
        == WHATSAPP_OUTREACH_PLAN_WORKFLOW_BINDING_TYPE
        and str(job.get("workflow_binding_id") or "").strip()
        and str(job.get("id") or job.get("job_id") or "").strip()
    }

    with _OUTREACH_STATE_LOCK:
        state = load_whatsapp_outreach_state()
        changed = False
        for plan_row in state["plans"]:
            plan_id = str(plan_row.get("plan_id") or "").strip()
            linked_cron_job_id = str(plan_row.get("linked_cron_job_id") or "").strip()
            if not linked_cron_job_id:
                continue
            if (plan_id, linked_cron_job_id) not in bound_pairs:
                plan_row["linked_cron_job_id"] = None
                active_targets = _active_targets_for_plan(state, plan_id)
                plan_row["trigger_mode"] = _determine_plan_trigger_mode(
                    len(active_targets), has_cron_binding=False
                )
                changed = True
        if changed:
            _write_whatsapp_outreach_state(state)


async def execute_whatsapp_approved_outreach(
    request: dict[str, Any] | None,
    *,
    authorized: bool,
    adapter: Any,
) -> dict[str, Any]:
    if not authorized:
        return _result(
            workflow_status="forbidden",
            founder_summary=(
                "WhatsApp approved outreach stayed blocked because this request "
                "was not owner-authorized."
            ),
            reason="owner authorization required",
        )

    try:
        normalized_request = normalize_whatsapp_approved_outreach_request(request)
    except ValueError as exc:
        return _result(
            workflow_status="invalid_request",
            founder_summary=(
                "WhatsApp approved outreach could not start because the "
                "instruction did not provide one exact target plus an "
                "operator objective."
            ),
            reason=str(exc),
        )

    bound_state = load_whatsapp_outreach_state()
    bound_request, _bound_plan, blocked_result = _resolve_bound_plan_request(
        normalized_request,
        state=bound_state,
    )
    if blocked_result is not None:
        return blocked_result
    normalized_request = bound_request or normalized_request

    approved_by_principal = "owner_operator"
    run_started_at_utc = utc_isoformat(utc_now())
    used_legacy_whatsapp_command_ingress = (
        normalized_request.get("operator_ingress_surface") is None
    )

    with _OUTREACH_STATE_LOCK:
        state = load_whatsapp_outreach_state()
        if (
            normalized_request.get("workflow_binding_type")
            == WHATSAPP_OUTREACH_PLAN_WORKFLOW_BINDING_TYPE
        ):
            plan_row, active_targets, founder_summary, reason = _validate_bound_plan(
                state,
                workflow_binding_id=str(normalized_request.get("workflow_binding_id")),
                trigger_reference_id=normalized_request.get("trigger_reference_id"),
            )
            if plan_row is None or active_targets is None:
                return _blocked_result(
                    founder_summary=str(founder_summary),
                    reason=str(reason),
                )
            plan_row = _ensure_plan_contract_fields(dict(plan_row))
            plan_targets = [dict(target) for target in active_targets]
        else:
            selected_fields = [
                field
                for field in WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS
                if normalized_request.get(field)
            ]
            approved_destination_chat_id = normalized_request.get(
                "approved_destination_chat_id"
            )
            if approved_destination_chat_id and selected_fields:
                return _result(
                    workflow_status="invalid_request",
                    founder_summary=(
                        "WhatsApp approved outreach could not start because the "
                        "instruction mixed preserved-thread selectors with a cold-start DM target."
                    ),
                    reason="approved outreach requires exactly one target basis",
                )
            if not approved_destination_chat_id and len(selected_fields) != 1:
                return _result(
                    workflow_status="invalid_request",
                    founder_summary=(
                        "WhatsApp approved outreach could not start because the "
                        "instruction did not provide one exact target plus an "
                        "operator objective."
                    ),
                    reason="approved outreach requires exactly one exact selector",
                )
            selector = {
                field: normalized_request.get(field)
                for field in WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS
            }
            target_resolution_source = (
                "approved_destination_chat_id"
                if approved_destination_chat_id
                else "preserved_history"
            )
            continuity_mode = (
                "cold_start_pending_first_send"
                if approved_destination_chat_id
                else "preserved_thread_follow_up"
            )
            plan_row, existing_target = _matching_plan_and_target(
                state,
                operator_objective=str(normalized_request["operator_objective"]),
                approved_by_principal=approved_by_principal,
                selector=selector,
                approved_destination_chat_id=approved_destination_chat_id,
            )
            approved_at_utc = utc_isoformat(utc_now())
            if plan_row is None or existing_target is None:
                plan_id = f"waplan-{uuid4()}"
                plan_target_id = f"watarget-{uuid4()}"
                plan_row = {
                    "plan_id": plan_id,
                    "plan_status": "active",
                    "trigger_mode": "instruction_only",
                    "communication_skill_name": (
                        WHATSAPP_DEFAULT_COMMUNICATION_SKILL_NAME
                    ),
                    "operator_objective": normalized_request["operator_objective"],
                    "created_by_session_id": None,
                    "operator_ingress_surface": normalized_request.get(
                        "operator_ingress_surface"
                    ),
                    "approved_by_principal": approved_by_principal,
                    "approved_at_utc": approved_at_utc,
                    "report_delivery_policy": "initiating_session",
                    "default_report_cadence": "end_of_run",
                    "linked_cron_job_id": None,
                    "used_legacy_whatsapp_command_ingress": (
                        used_legacy_whatsapp_command_ingress
                    ),
                }
                _ensure_plan_contract_fields(plan_row)
                plan_targets = [
                    {
                        "plan_target_id": plan_target_id,
                        "plan_id": plan_id,
                        "target_status": "active",
                        **selector,
                        "approved_destination_chat_id": approved_destination_chat_id,
                        "target_resolution_source": target_resolution_source,
                        "continuity_mode": continuity_mode,
                        "target_objective_override": None,
                        "max_outbound_messages_per_run": 1,
                        "last_resolved_conversation_key": None,
                        "last_observed_message_at_utc": None,
                    }
                ]
            else:
                plan_row = _ensure_plan_contract_fields(dict(plan_row))
                plan_row["plan_status"] = "active"
                if normalized_request.get("operator_ingress_surface"):
                    plan_row["operator_ingress_surface"] = normalized_request.get(
                        "operator_ingress_surface"
                    )
                    plan_row["used_legacy_whatsapp_command_ingress"] = False
                elif "used_legacy_whatsapp_command_ingress" not in plan_row:
                    plan_row["used_legacy_whatsapp_command_ingress"] = True
                plan_row["communication_skill_name"] = (
                    _normalized_communication_skill_name(
                        plan_row.get("communication_skill_name")
                    )
                )
                plan_row["approved_at_utc"] = (
                    plan_row.get("approved_at_utc") or approved_at_utc
                )
                plan_targets = [dict(existing_target)]

        plan_id = str(plan_row["plan_id"])
        plan_row["communication_skill_name"] = _normalized_communication_skill_name(
            plan_row.get("communication_skill_name")
        )
        plan_row["trigger_mode"] = _determine_plan_trigger_mode(
            len(plan_targets),
            has_cron_binding=bool(plan_row.get("linked_cron_job_id")),
        )
        run_id = f"warun-{uuid4()}"
        run = {
            "run_id": run_id,
            "plan_id": plan_id,
            "run_status": "running",
            "workflow_binding_type": normalized_request.get("workflow_binding_type"),
            "workflow_binding_id": normalized_request.get("workflow_binding_id"),
            "trigger_source": normalized_request["trigger_source"],
            "trigger_reference_id": normalized_request.get("trigger_reference_id"),
            "operator_ingress_surface": normalized_request.get(
                "operator_ingress_surface"
            ),
            "run_started_at_utc": run_started_at_utc,
            "run_completed_at_utc": None,
            "target_count": len(plan_targets),
            "completed_target_count": 0,
            "failed_target_count": 0,
            "report_delivery_target": normalized_request.get("report_delivery_target"),
            "report_id": None,
            "used_legacy_whatsapp_command_ingress": (
                used_legacy_whatsapp_command_ingress
            ),
        }
        _ensure_run_contract_fields(run, plan=plan_row)
        executions: list[dict[str, Any]] = []
        state["plans"] = [
            row for row in state["plans"] if row.get("plan_id") != plan_id
        ]
        state["plans"].append(dict(plan_row))
        for plan_target_row in plan_targets:
            state["plan_targets"] = [
                row
                for row in state["plan_targets"]
                if row.get("plan_target_id") != plan_target_row["plan_target_id"]
            ]
            state["plan_targets"].append(dict(plan_target_row))
            executions.append({
                "target_execution_id": f"waexec-{uuid4()}",
                "run_id": run_id,
                "plan_target_id": plan_target_row["plan_target_id"],
                "execution_status": "queued",
                "resolved_conversation_key": None,
                "resolved_destination_key": None,
                "resolved_destination_chat_id": None,
                "continuity_mode": plan_target_row.get("continuity_mode"),
                "target_resolution_source": plan_target_row.get(
                    "target_resolution_source"
                ),
                "approved_destination_chat_id_snapshot": None,
                "had_prior_preserved_thread": True,
                "group_chat_id": plan_target_row.get("group_chat_id"),
                "dm_counterparty_id": plan_target_row.get("dm_counterparty_id"),
                "destination_context_type": None,
                "destination_target_id": None,
                "history_window_start_utc": None,
                "history_window_end_utc": None,
                "history_record_count": 0,
                "communication_skill_name": plan_row["communication_skill_name"],
                "communication_stage": None,
                "message_generation_mode": None,
                "draft_outcome": None,
                "outbound_message_text": None,
                "handoff_reason": None,
                "dispatch_group_id": None,
                "message_id": None,
                "last_error": None,
            })
            _ensure_execution_contract_fields(executions[-1])
        state["runs"] = [row for row in state["runs"] if row.get("run_id") != run_id]
        state["runs"].append(dict(run))
        for execution in executions:
            state["target_executions"] = [
                row
                for row in state["target_executions"]
                if row.get("target_execution_id") != execution["target_execution_id"]
            ]
            state["target_executions"].append(dict(execution))
        _write_whatsapp_outreach_state(state)

    send_callable = _resolve_send_callable(adapter)
    target_rows: list[dict[str, Any]] = []
    report_uncertainties: list[str] = []
    failed_count = 0
    completed_count = 0
    run_window_start_utc: str | None = None
    run_window_end_utc: str | None = None

    for plan_target_row, execution in zip(plan_targets, executions, strict=False):
        selector = _exact_selector_from_row(plan_target_row)
        selected_fields = [field for field, value in selector.items() if value]
        approved_destination_chat_id = _json_safe_string(
            plan_target_row.get("approved_destination_chat_id")
        )
        target_resolution_source = (
            _json_safe_string(plan_target_row.get("target_resolution_source"))
            or "preserved_history"
        )
        continuity_mode = _json_safe_string(plan_target_row.get("continuity_mode"))
        if target_resolution_source == "approved_destination_chat_id":
            destination_fields = build_whatsapp_destination_fields(
                str(approved_destination_chat_id or "")
            )
            cold_start_error = _cold_start_failure_reason(
                approved_destination_chat_id=approved_destination_chat_id,
                destination_fields=destination_fields,
            )
            if cold_start_error:
                execution.update({
                    "execution_status": "blocked",
                    "last_error": cold_start_error,
                    "continuity_mode": continuity_mode
                    or "cold_start_pending_first_send",
                    "target_resolution_source": "approved_destination_chat_id",
                    "approved_destination_chat_id_snapshot": approved_destination_chat_id,
                    "had_prior_preserved_thread": False,
                })
                target_rows.append({
                    "plan_target_id": plan_target_row["plan_target_id"],
                    "resolved_target": {
                        "conversation_key": None,
                        "destination_key": destination_fields.get("destination_key"),
                        "group_chat_id": None,
                        "dm_counterparty_id": destination_fields.get(
                            "dm_counterparty_id"
                        ),
                        "destination_context_type": destination_fields.get(
                            "destination_context_type"
                        ),
                        "destination_chat_id": destination_fields.get(
                            "destination_chat_id"
                        ),
                        "destination_target_id": destination_fields.get(
                            "destination_target_id"
                        ),
                    },
                    "continuity_mode": "cold_start_pending_first_send",
                    "had_prior_preserved_thread": False,
                    "observable_status": "unresolved",
                    "status_basis": "direct_conversation_evidence",
                    "latest_observed_message_at_utc": None,
                    "open_items": [],
                    "handoff_reason": None,
                    "uncertainties": [cold_start_error],
                })
                _ensure_report_row_contract_fields(target_rows[-1])
                failed_count += 1
                completed_count += 1
                continue
            resolved_target = {
                "conversation_key": None,
                "destination_key": destination_fields.get("destination_key"),
                "group_chat_id": None,
                "dm_counterparty_id": destination_fields.get("dm_counterparty_id"),
                "destination_context_type": destination_fields.get(
                    "destination_context_type"
                ),
                "destination_chat_id": destination_fields.get("destination_chat_id"),
                "destination_target_id": destination_fields.get(
                    "destination_target_id"
                ),
            }
            continuity_rows = []
            history_window_start_utc = utc_isoformat(utc_now())
            history_window_end_utc = history_window_start_utc
            resolution = {
                "resolution_status": "resolved",
                "resolved_target": resolved_target,
            }
        elif len(selected_fields) == 1:
            resolution = ResolveWhatsAppExactTarget(selector)
        else:
            resolution = _resolve_target_from_preserved_history(selector)
        resolved_target = resolution.get("resolved_target")
        if resolution.get("resolution_status") != "resolved" or not isinstance(
            resolved_target, dict
        ):
            execution["execution_status"] = "blocked"
            execution["last_error"] = (
                "exact target could not be resolved from preserved history"
            )
            target_rows.append({
                "plan_target_id": plan_target_row["plan_target_id"],
                "resolved_target": {
                    **selector,
                    "destination_context_type": None,
                    "destination_chat_id": None,
                    "destination_target_id": None,
                },
                "observable_status": "unresolved",
                "status_basis": "direct_conversation_evidence",
                "latest_observed_message_at_utc": None,
                "open_items": [],
                "uncertainties": [
                    "exact target could not be resolved from preserved history"
                ],
            })
            _ensure_report_row_contract_fields(target_rows[-1])
            failed_count += 1
            completed_count += 1
            continue
        if target_resolution_source != "approved_destination_chat_id":
            continuity_rows = query_whatsapp_records_any_time(
                conversation_key=resolved_target.get("conversation_key"),
                destination_key=resolved_target.get("destination_key"),
                destination_context_type=resolved_target.get(
                    "destination_context_type"
                ),
                group_chat_id=resolved_target.get("group_chat_id"),
                dm_counterparty_id=resolved_target.get("dm_counterparty_id"),
            )
            history_window_start_utc = (
                continuity_rows[0].get("effective_event_at_utc")
                if continuity_rows
                else None
            )
            history_window_end_utc = (
                continuity_rows[-1].get("effective_event_at_utc")
                if continuity_rows
                else None
            )
            if history_window_start_utc is None:
                history_window_start_utc = utc_isoformat(utc_now())
            if history_window_end_utc is None:
                history_window_end_utc = history_window_start_utc
        if (
            run_window_start_utc is None
            or history_window_start_utc < run_window_start_utc
        ):
            run_window_start_utc = history_window_start_utc
        if run_window_end_utc is None or history_window_end_utc > run_window_end_utc:
            run_window_end_utc = history_window_end_utc

        plan_target_row.update({
            **_exact_selector_from_row(resolved_target),
            "approved_destination_chat_id": approved_destination_chat_id,
            "target_resolution_source": target_resolution_source,
            "continuity_mode": continuity_mode
            or (
                "cold_start_pending_first_send"
                if target_resolution_source == "approved_destination_chat_id"
                else "preserved_thread_follow_up"
            ),
            "last_resolved_conversation_key": resolved_target.get("conversation_key"),
            "last_observed_message_at_utc": history_window_end_utc,
        })
        execution.update({
            "execution_status": "resolved",
            "resolved_conversation_key": resolved_target.get("conversation_key"),
            "resolved_destination_key": resolved_target.get("destination_key"),
            "resolved_destination_chat_id": resolved_target.get("destination_chat_id"),
            "group_chat_id": resolved_target.get("group_chat_id"),
            "dm_counterparty_id": resolved_target.get("dm_counterparty_id"),
            "destination_context_type": resolved_target.get("destination_context_type"),
            "destination_target_id": resolved_target.get("destination_target_id"),
            "history_window_start_utc": history_window_start_utc,
            "history_window_end_utc": history_window_end_utc,
            "history_record_count": len(continuity_rows),
            "continuity_mode": continuity_mode
            or (
                "cold_start_pending_first_send"
                if target_resolution_source == "approved_destination_chat_id"
                else "preserved_thread_follow_up"
            ),
            "target_resolution_source": target_resolution_source,
            "approved_destination_chat_id_snapshot": approved_destination_chat_id,
            "had_prior_preserved_thread": target_resolution_source
            != "approved_destination_chat_id",
            "communication_skill_name": plan_row["communication_skill_name"],
            "last_error": None,
        })
        _ensure_execution_contract_fields(execution)

        outbound_message_text = normalized_request["message_text"]
        execution["communication_stage"] = _select_communication_stage(continuity_rows)
        if normalized_request["message_text"] is not None:
            execution["message_generation_mode"] = "operator_text_override"
            execution["draft_outcome"] = "send_message"
            execution["outbound_message_text"] = outbound_message_text
        else:
            try:
                draft_request, draft = _draft_whatsapp_message_via_skill(
                    communication_skill_name=plan_row["communication_skill_name"],
                    operator_objective=str(plan_row.get("operator_objective") or ""),
                    resolved_target=resolved_target,
                    history_rows=continuity_rows,
                    message_text_override=normalized_request["message_text"],
                    trigger_source=str(normalized_request["trigger_source"]),
                    task_id=run_id,
                )
                execution["communication_skill_name"] = draft_request[
                    "communication_skill_name"
                ]
                execution["communication_stage"] = draft_request["communication_stage"]
                execution["message_generation_mode"] = "skill_generated"
                execution["draft_outcome"] = draft["draft_outcome"]
                execution["outbound_message_text"] = draft["outbound_message_text"]
                execution["handoff_reason"] = draft["handoff_reason"]
                if draft["draft_outcome"] == "send_message":
                    outbound_message_text = draft["outbound_message_text"]
                elif draft["draft_outcome"] == "no_send_required":
                    outbound_message_text = None
                    execution["execution_status"] = "no_send_required"
                else:
                    outbound_message_text = None
                    execution["execution_status"] = "handoff_required"
            except Exception as exc:
                execution["execution_status"] = "blocked"
                execution["last_error"] = _json_safe_string(str(exc))

        if execution.get("execution_status") not in {
            "blocked",
            "no_send_required",
            "handoff_required",
        }:
            if send_callable is None:
                execution["execution_status"] = "blocked"
                execution["last_error"] = "WhatsApp adapter unavailable"
            else:
                send_result = await send_callable(
                    str(resolved_target["destination_chat_id"]),
                    str(outbound_message_text),
                )
                raw_response = _safe_instance_attr(send_result, "raw_response") or {}
                if isinstance(raw_response, dict):
                    execution["dispatch_group_id"] = _json_safe_string(
                        raw_response.get("dispatch_group_id")
                    )
                execution["message_id"] = _json_safe_string(
                    _safe_instance_attr(send_result, "message_id")
                )
                if _safe_instance_attr(send_result, "success") is True:
                    execution["execution_status"] = "sent"
                else:
                    execution["execution_status"] = "send_failed"
                    execution["last_error"] = _json_safe_string(
                        _safe_instance_attr(send_result, "error")
                    )

        if (
            execution.get("execution_status") == "sent"
            and execution.get("target_resolution_source")
            == "approved_destination_chat_id"
            and execution.get("outbound_message_text")
        ):
            cold_start_event_at_utc = utc_isoformat(utc_now())
            _persist_cold_start_outbound_record(
                approved_destination_chat_id_snapshot=str(
                    execution.get("approved_destination_chat_id_snapshot")
                ),
                outbound_message_text=str(execution.get("outbound_message_text")),
                message_id=_json_safe_string(execution.get("message_id")),
                effective_event_at_utc=str(cold_start_event_at_utc),
            )
            persisted_fields = build_whatsapp_destination_fields(
                str(execution.get("approved_destination_chat_id_snapshot") or "")
            )
            plan_target_row.update({
                **_exact_selector_from_row({
                    "conversation_key": persisted_fields.get("destination_key"),
                    "destination_key": persisted_fields.get("destination_key"),
                    "group_chat_id": None,
                    "dm_counterparty_id": persisted_fields.get("dm_counterparty_id"),
                }),
                "target_resolution_source": "preserved_history",
                "continuity_mode": "preserved_thread_follow_up",
                "last_resolved_conversation_key": persisted_fields.get(
                    "destination_key"
                ),
                "last_observed_message_at_utc": cold_start_event_at_utc,
            })
            execution.update({
                "resolved_conversation_key": persisted_fields.get("destination_key"),
                "resolved_destination_key": persisted_fields.get("destination_key"),
                "resolved_destination_chat_id": persisted_fields.get(
                    "destination_chat_id"
                ),
                "dm_counterparty_id": persisted_fields.get("dm_counterparty_id"),
                "destination_context_type": persisted_fields.get(
                    "destination_context_type"
                ),
                "destination_target_id": persisted_fields.get("destination_target_id"),
                "history_window_start_utc": cold_start_event_at_utc,
                "history_window_end_utc": cold_start_event_at_utc,
                "history_record_count": 0,
                "continuity_mode": "cold_start_first_send_completed",
                "target_resolution_source": "approved_destination_chat_id",
                "had_prior_preserved_thread": False,
            })
            resolved_target = {
                "conversation_key": persisted_fields.get("destination_key"),
                "destination_key": persisted_fields.get("destination_key"),
                "group_chat_id": None,
                "dm_counterparty_id": persisted_fields.get("dm_counterparty_id"),
                "destination_context_type": persisted_fields.get(
                    "destination_context_type"
                ),
                "destination_chat_id": persisted_fields.get("destination_chat_id"),
                "destination_target_id": persisted_fields.get("destination_target_id"),
            }
            history_window_start_utc = cold_start_event_at_utc
            history_window_end_utc = cold_start_event_at_utc
            if (
                run_window_start_utc is None
                or history_window_start_utc < run_window_start_utc
            ):
                run_window_start_utc = history_window_start_utc
            if (
                run_window_end_utc is None
                or history_window_end_utc > run_window_end_utc
            ):
                run_window_end_utc = history_window_end_utc

        observable_status = _observable_status_from_history_and_execution(
            continuity_rows,
            str(execution.get("execution_status") or ""),
        )
        row_uncertainties: list[str] = []
        if execution.get("execution_status") == "sent":
            row_uncertainties.append(
                "observable status is bounded to preserved history before the next reply arrives"
            )
            if execution.get("had_prior_preserved_thread") is False:
                row_uncertainties.append(
                    "execution began without prior preserved thread context"
                )
        if execution.get("execution_status") == "handoff_required" and execution.get(
            "handoff_reason"
        ):
            row_uncertainties.append(str(execution["handoff_reason"]))
        if execution.get("last_error"):
            row_uncertainties.append(str(execution["last_error"]))
        target_rows.append({
            "plan_target_id": plan_target_row["plan_target_id"],
            "resolved_target": resolved_target,
            "continuity_mode": execution.get("continuity_mode"),
            "had_prior_preserved_thread": execution.get("had_prior_preserved_thread"),
            "observable_status": observable_status,
            "status_basis": _status_basis_for_observable_status(observable_status),
            "latest_observed_message_at_utc": history_window_end_utc,
            "open_items": [],
            "handoff_reason": execution.get("handoff_reason"),
            "uncertainties": row_uncertainties,
        })
        _ensure_report_row_contract_fields(target_rows[-1])
        if execution["execution_status"] in {"blocked", "send_failed"}:
            failed_count += 1
        completed_count += 1

    if run_window_start_utc is None:
        run_window_start_utc = utc_isoformat(utc_now())
    if run_window_end_utc is None:
        run_window_end_utc = run_window_start_utc

    success_count = sum(
        1
        for item in executions
        if item.get("execution_status") in {"sent", "no_send_required"}
    )
    blocked_or_handoff_count = sum(
        1
        for item in executions
        if item.get("execution_status") in {"blocked", "handoff_required"}
    )
    send_failed_count = sum(
        1 for item in executions if item.get("execution_status") == "send_failed"
    )

    if not target_rows:
        run["run_status"] = "blocked"
        founder_summary = "WhatsApp approved outreach stayed blocked because no executable approved targets were available."
        report_status = "no_targets"
    elif success_count and failed_count == 0:
        run["run_status"] = "completed"
        founder_summary = (
            f"WhatsApp approved outreach executed {completed_count} approved target"
            f"{'s' if completed_count != 1 else ''} through isolated single-target steps and produced an evidence-bounded run report."
        )
        report_status = "ready"
    elif success_count and failed_count:
        run["run_status"] = "completed_with_failures"
        founder_summary = (
            f"WhatsApp approved outreach executed {completed_count} approved targets with {failed_count} isolated failure"
            f"{'s' if failed_count != 1 else ''}; the report stays bounded to preserved history plus explicit run metadata."
        )
        report_status = "partial"
    elif blocked_or_handoff_count:
        run["run_status"] = "blocked"
        founder_summary = "WhatsApp approved outreach stayed blocked because none of the approved targets reached a send or no-send completion state."
        report_status = "failed"
    elif send_failed_count == len(executions):
        run["run_status"] = "failed"
        founder_summary = "WhatsApp approved outreach attempted the approved target set but every isolated target execution failed at send time."
        report_status = "failed"
    else:
        run["run_status"] = "failed"
        founder_summary = "WhatsApp approved outreach attempted the approved target set but every isolated target execution failed or blocked."
        report_status = "failed"

    founder_summary = (
        f"{_founder_visible_ingress_summary(operator_ingress_surface=run.get('operator_ingress_surface'), interaction_lane=run.get('interaction_lane'), used_legacy_whatsapp_command_ingress=bool(run.get('used_legacy_whatsapp_command_ingress')))} "
        f"{founder_summary}"
    )

    if failed_count:
        report_uncertainties.append(
            "one or more targets failed or blocked; review per-target outcome rows"
        )

    run["run_completed_at_utc"] = utc_isoformat(utc_now())
    run["completed_target_count"] = completed_count
    run["failed_target_count"] = failed_count

    report = {
        "report_id": f"wareport-{uuid4()}",
        "plan_id": plan_id,
        "run_id": run_id,
        "report_status": report_status,
        "report_window_start_utc": run_window_start_utc,
        "report_window_end_utc": run_window_end_utc,
        "report_text": founder_summary,
        "target_rows": target_rows,
        "uncertainties": report_uncertainties,
    }
    _ensure_report_contract_fields(report)
    run["report_id"] = report["report_id"]

    with _OUTREACH_STATE_LOCK:
        state = load_whatsapp_outreach_state()
        persisted_plan = _find_first(state["plans"], "plan_id", plan_id)
        if persisted_plan is None:
            state["plans"].append(dict(plan_row))
            persisted_plan = state["plans"][-1]
        else:
            persisted_plan.update(dict(plan_row))
        for plan_target_row in plan_targets:
            persisted_target = _find_first(
                state["plan_targets"],
                "plan_target_id",
                plan_target_row["plan_target_id"],
            )
            if persisted_target is None:
                state["plan_targets"].append(dict(plan_target_row))
            else:
                persisted_target.update(dict(plan_target_row))
        persisted_run = _find_first(state["runs"], "run_id", run_id)
        if persisted_run is None:
            state["runs"].append(dict(run))
            persisted_run = state["runs"][-1]
        else:
            persisted_run.update(dict(run))
        for execution in executions:
            persisted_execution = _find_first(
                state["target_executions"],
                "target_execution_id",
                execution["target_execution_id"],
            )
            if persisted_execution is None:
                state["target_executions"].append(dict(execution))
            else:
                persisted_execution.update(dict(execution))
        state["reports"] = [
            row
            for row in state["reports"]
            if row.get("report_id") != report["report_id"]
        ]
        state["reports"].append(dict(report))
        _write_whatsapp_outreach_state(state)

        plan = _enriched_plan(state, persisted_plan) or {}
        run = _enriched_run(state, persisted_run) or {}
        execution = run.get("target_executions", [None])[0]

    first_error = next(
        (str(item.get("last_error")) for item in executions if item.get("last_error")),
        None,
    )
    return _result(
        workflow_status="ready",
        plan=plan,
        run=run,
        execution=execution,
        founder_summary=founder_summary,
        reason=first_error,
    )


def format_whatsapp_approved_outreach_result(result: dict[str, Any]) -> str:
    lines = [
        "WhatsApp approved outreach",
        f"Status: {result.get('workflow_status', 'invalid_request')}",
    ]
    workflow_status = result.get("workflow_status")
    if workflow_status != "ready":
        if result.get("reason"):
            lines.append(f"Reason: {result['reason']}")
        if result.get("founder_summary"):
            lines.append(result["founder_summary"])
        return "\n".join(lines)

    plan = result.get("plan") or {}
    run = result.get("run") or {}
    target_executions = run.get("target_executions") or []
    execution = (
        result.get("execution")
        or (target_executions[0] if target_executions else {})
        or {}
    )
    resolved_target = execution.get("resolved_target") or {}
    report = next(
        (
            row
            for row in load_whatsapp_outreach_state().get("reports", [])
            if row.get("report_id") == run.get("report_id")
        ),
        None,
    )

    lines.extend([
        f"plan_id: {plan.get('plan_id')}",
        f"communication_skill_name: {plan.get('communication_skill_name')}",
        f"conversation_channel_mode: {plan.get('conversation_channel_mode')}",
        f"operator_governance_policy: {plan.get('operator_governance_policy')}",
        f"run_id: {run.get('run_id')}",
        f"workflow_binding_type: {run.get('workflow_binding_type')}",
        f"workflow_binding_id: {run.get('workflow_binding_id')}",
        f"trigger_source: {run.get('trigger_source')}",
        f"trigger_reference_id: {run.get('trigger_reference_id')}",
        f"operator_ingress_surface: {run.get('operator_ingress_surface')}",
        f"governance_interaction_lane: {run.get('interaction_lane')}",
        f"used_legacy_whatsapp_command_ingress: {run.get('used_legacy_whatsapp_command_ingress')}",
        f"operator_objective: {plan.get('operator_objective')}",
        f"run_status: {run.get('run_status')}",
        f"target_count: {run.get('target_count')}",
        f"completed_target_count: {run.get('completed_target_count')}",
        f"failed_target_count: {run.get('failed_target_count')}",
        f"message_generation_mode: {execution.get('message_generation_mode')}",
        f"communication_stage: {execution.get('communication_stage')}",
        f"draft_outcome: {execution.get('draft_outcome')}",
        f"execution_status: {execution.get('execution_status')}",
        f"external_interaction_lane: {execution.get('interaction_lane')}",
        f"continuity_mode: {execution.get('continuity_mode')}",
        f"target_resolution_source: {execution.get('target_resolution_source')}",
        f"approved_destination_chat_id_snapshot: {execution.get('approved_destination_chat_id_snapshot')}",
        f"had_prior_preserved_thread: {execution.get('had_prior_preserved_thread')}",
        f"destination_chat_id: {resolved_target.get('destination_chat_id')}",
        f"conversation_key: {resolved_target.get('conversation_key')}",
        f"destination_key: {resolved_target.get('destination_key')}",
        f"group_chat_id: {resolved_target.get('group_chat_id')}",
        f"dm_counterparty_id: {resolved_target.get('dm_counterparty_id')}",
        f"history_window_start_utc: {execution.get('history_window_start_utc')}",
        f"history_window_end_utc: {execution.get('history_window_end_utc')}",
        f"history_record_count: {execution.get('history_record_count')}",
        f"dispatch_group_id: {execution.get('dispatch_group_id')}",
        f"message_id: {execution.get('message_id')}",
    ])
    if execution.get("outbound_message_text"):
        lines.append(f"outbound_message_text: {execution.get('outbound_message_text')}")
    if execution.get("handoff_reason"):
        lines.append(f"handoff_reason: {execution.get('handoff_reason')}")
    if execution.get("last_error"):
        lines.append(f"last_error: {execution.get('last_error')}")
    if isinstance(report, dict):
        lines.extend([
            f"report_id: {report.get('report_id')}",
            f"report_status: {report.get('report_status')}",
            f"report_interaction_lane: {report.get('interaction_lane')}",
            f"report_target_rows: {len(report.get('target_rows') or [])}",
        ])
        first_report_row = (report.get("target_rows") or [None])[0] or {}
        lines.extend([
            f"report_continuity_mode: {first_report_row.get('continuity_mode')}",
            f"report_had_prior_preserved_thread: {first_report_row.get('had_prior_preserved_thread')}",
            f"report_row_interaction_lane: {first_report_row.get('interaction_lane')}",
        ])
    if target_executions:
        lines.extend(["", "Target executions:"])
        lines.extend(_execution_summary_line(item) for item in target_executions)
    if result.get("founder_summary"):
        lines.extend(["", result["founder_summary"]])
    return "\n".join(lines)


__all__ = [
    "WHATSAPP_ALLOWED_INTERACTION_LANES",
    "WHATSAPP_ALLOWED_OPERATOR_INGRESS_SURFACES",
    "WHATSAPP_CONVERSATION_CHANNEL_MODE",
    "WHATSAPP_OPERATOR_GOVERNANCE_POLICY",
    "WHATSAPP_APPROVED_OUTREACH_ALLOWED_FIELDS",
    "WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS",
    "WHATSAPP_OUTREACH_PLAN_WORKFLOW_BINDING_TYPE",
    "WHATSAPP_OUTREACH_STATE_SCHEMA_VERSION",
    "append_whatsapp_outreach_run_record",
    "build_cli_chat_whatsapp_outreach_requests",
    "build_local_chat_whatsapp_outreach_requests",
    "bind_whatsapp_outreach_plan_to_cron_job",
    "execute_whatsapp_approved_outreach",
    "format_whatsapp_approved_outreach_result",
    "is_whatsapp_approved_outreach_instruction",
    "is_whatsapp_legacy_outreach_instruction",
    "is_whatsapp_local_governance_instruction",
    "load_whatsapp_outreach_state",
    "load_whatsapp_outreach_run_records",
    "normalize_whatsapp_approved_outreach_request",
    "parse_whatsapp_approved_outreach_instruction",
    "sync_whatsapp_outreach_plan_cron_bindings",
    "unbind_whatsapp_outreach_plan_from_cron_job",
]
