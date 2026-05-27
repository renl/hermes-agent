from __future__ import annotations

import sys

import pytest


def test_cli_whatsapp_cold_start_validation_prepare_invokes_workflow(
    monkeypatch, capsys
):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_prepare(request, *, authorized, created_by_principal):
        captured["request"] = dict(request)
        captured["authorized"] = authorized
        captured["created_by_principal"] = created_by_principal
        return {
            "workflow_status": "prepared",
            "continuity_scope": {
                "continuity_scope_id": "wascope-1",
                "continuity_scope_kind": "cold_start_validation",
                "cold_start_validation_mode": "non_destructive_isolation",
                "plan_target_id": "watarget-1",
                "target_destination_key": "whatsapp:dm:15551230000",
                "target_dm_counterparty_id": "15551230000",
                "approved_destination_chat_id_snapshot": "15551230000@s.whatsapp.net",
                "validation_prepare_surface": "cli_command",
                "created_by_principal": "owner_operator",
                "operator_reason": "prepare exact target",
                "created_at_utc": "2024-06-02T09:00:00Z",
            },
            "founder_summary": "Prepared non-destructive validation scope.",
        }

    monkeypatch.setattr(
        "gateway.whatsapp_approved_outreach.prepare_whatsapp_cold_start_validation",
        fake_prepare,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "whatsapp",
            "cold-start-validation",
            "prepare",
            "--plan-id",
            "waplan-1",
            "--dm-counterparty-id",
            "15551230000",
            "--operator-reason",
            "prepare exact target",
            "--approved-destination-chat-id",
            "15551230000@s.whatsapp.net",
        ],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert captured == {
        "request": {
            "plan_id": "waplan-1",
            "dm_counterparty_id": "15551230000",
            "cold_start_validation_mode": "non_destructive_isolation",
            "operator_reason": "prepare exact target",
            "approved_destination_chat_id": "15551230000@s.whatsapp.net",
            "validation_prepare_surface": "cli_command",
        },
        "authorized": True,
        "created_by_principal": "owner_operator",
    }
    assert "WhatsApp cold-start validation" in output
    assert "Status: prepared" in output
    assert "continuity_scope_id: wascope-1" in output
    assert "cold_start_validation_mode: non_destructive_isolation" in output
    assert "validation_prepare_surface: cli_command" in output
    assert "Prepared non-destructive validation scope." in output


def test_cli_whatsapp_cold_start_validation_prepare_defaults_mode(monkeypatch, capsys):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_prepare(request, *, authorized, created_by_principal):
        captured.update(request)
        return {
            "workflow_status": "prepared",
            "continuity_scope": {"continuity_scope_id": "wascope-2"},
        }

    monkeypatch.setattr(
        "gateway.whatsapp_approved_outreach.prepare_whatsapp_cold_start_validation",
        fake_prepare,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "whatsapp",
            "cold-start-validation",
            "prepare",
            "--plan-id",
            "waplan-1",
            "--dm-counterparty-id",
            "15551230000",
            "--operator-reason",
            "prepare exact target",
        ],
    )

    main_mod.main()

    _ = capsys.readouterr()
    assert captured["cold_start_validation_mode"] == "non_destructive_isolation"


def test_cli_whatsapp_cold_start_validation_prepare_returns_non_zero_on_failure(
    monkeypatch, capsys
):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(
        "gateway.whatsapp_approved_outreach.prepare_whatsapp_cold_start_validation",
        lambda request, **_kwargs: {
            "workflow_status": "invalid_request",
            "reason": "plan_id is required",
            "founder_summary": "Preparation failed.",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "whatsapp",
            "cold-start-validation",
            "prepare",
            "--plan-id",
            "waplan-1",
            "--dm-counterparty-id",
            "15551230000",
            "--operator-reason",
            "prepare exact target",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    output = capsys.readouterr().out
    assert exc.value.code == 1
    assert "Status: invalid_request" in output
    assert "Reason: plan_id is required" in output


def test_cli_whatsapp_cold_start_validation_prepare_requires_prepare_subcommand(
    monkeypatch, capsys
):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "whatsapp", "cold-start-validation"],
    )

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    stderr = capsys.readouterr().err
    assert exc.value.code == 1
    assert "usage: hermes whatsapp cold-start-validation prepare" in stderr
