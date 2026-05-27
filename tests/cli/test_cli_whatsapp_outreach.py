from __future__ import annotations

import asyncio
from types import SimpleNamespace

from cli import HermesCLI


def test_cli_whatsapp_outreach_keeps_adapter_lifecycle_on_one_event_loop(monkeypatch):
    captured = {}

    class LoopBoundAdapter:
        def __init__(self, _config):
            self.connect_loop = None
            self.execute_loop = None
            self.disconnect_loop = None

        async def connect(self):
            self.connect_loop = asyncio.get_running_loop()
            captured["adapter"] = self
            return True

        async def disconnect(self):
            self.disconnect_loop = asyncio.get_running_loop()

    async def _fake_execute(run_request, *, authorized, adapter):
        assert authorized is True
        adapter.execute_loop = asyncio.get_running_loop()
        assert adapter.execute_loop is adapter.connect_loop
        assert run_request["operator_ingress_surface"] == "cli_chat"
        return {"workflow_status": "ready"}

    class _Console:
        def print(self, *_args, **_kwargs):
            return None

    from gateway.config import Platform

    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: SimpleNamespace(
            platforms={Platform.WHATSAPP: SimpleNamespace(enabled=True)}
        ),
    )
    monkeypatch.setattr("gateway.platforms.whatsapp.WhatsAppAdapter", LoopBoundAdapter)
    monkeypatch.setattr("cli.execute_whatsapp_approved_outreach", _fake_execute)
    monkeypatch.setattr(
        "cli.format_whatsapp_approved_outreach_result",
        lambda result: f"Status: {result['workflow_status']}",
    )
    monkeypatch.setattr("cli.ChatConsole", lambda: _Console())
    monkeypatch.setattr(
        "cli._render_final_assistant_content",
        lambda text, mode=None: text,
    )

    hermes_cli = HermesCLI.__new__(HermesCLI)
    hermes_cli.session_id = "sess-cli-runtime"
    hermes_cli.conversation_history = []
    hermes_cli.final_response_markdown = "strip"
    hermes_cli._print_user_message_preview = lambda _text: None
    hermes_cli._scrollback_box_width = lambda: 80

    handled = hermes_cli._run_whatsapp_cli_chat_outreach(
        "whatsapp outreach approved_destination_chat_id=15551230000@s.whatsapp.net "
        'operator_objective="Request the first quote" '
        'message_text="Hello from Hermes."'
    )

    assert handled is True
    adapter = captured["adapter"]
    assert adapter.connect_loop is not None
    assert adapter.execute_loop is adapter.connect_loop
    assert adapter.disconnect_loop is adapter.connect_loop
    assert hermes_cli.conversation_history[0]["role"] == "user"
    assert hermes_cli.conversation_history[1]["content"] == "Status: ready"


def test_cli_whatsapp_local_governance_alias_uses_same_execution_path(monkeypatch):
    captured = {}

    class LoopBoundAdapter:
        def __init__(self, _config):
            return None

        async def connect(self):
            return True

        async def disconnect(self):
            return None

    async def _fake_execute(run_request, *, authorized, adapter):
        assert authorized is True
        captured["run_request"] = run_request
        return {"workflow_status": "ready"}

    class _Console:
        def print(self, *_args, **_kwargs):
            return None

    from gateway.config import Platform

    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: SimpleNamespace(
            platforms={Platform.WHATSAPP: SimpleNamespace(enabled=True)}
        ),
    )
    monkeypatch.setattr("gateway.platforms.whatsapp.WhatsAppAdapter", LoopBoundAdapter)
    monkeypatch.setattr("cli.execute_whatsapp_approved_outreach", _fake_execute)
    monkeypatch.setattr(
        "cli.format_whatsapp_approved_outreach_result",
        lambda result: f"Status: {result['workflow_status']}",
    )
    monkeypatch.setattr("cli.ChatConsole", lambda: _Console())
    monkeypatch.setattr(
        "cli._render_final_assistant_content",
        lambda text, mode=None: text,
    )

    hermes_cli = HermesCLI.__new__(HermesCLI)
    hermes_cli.session_id = "sess-cli-runtime"
    hermes_cli.conversation_history = []
    hermes_cli.final_response_markdown = "strip"
    hermes_cli._print_user_message_preview = lambda _text: None
    hermes_cli._scrollback_box_width = lambda: 80

    handled = hermes_cli._run_whatsapp_local_governance_ingress(
        "whatsapp start approved_destination_chat_id=15551230000@s.whatsapp.net "
        'operator_objective="Request the first quote" '
        'message_text="Hello from Hermes."'
    )

    assert handled is True
    assert captured["run_request"]["operator_ingress_surface"] == "cli_chat"
    assert captured["run_request"]["interaction_lane"] == "operator_governance"
    assert (
        captured["run_request"]["conversation_channel_mode"] == "conversational_primary"
    )
    assert (
        captured["run_request"]["operator_governance_policy"]
        == "owner_approved_conversation"
    )


def test_cli_chat_routes_local_governance_before_agent_turn(monkeypatch):
    captured = {}

    hermes_cli = HermesCLI.__new__(HermesCLI)
    hermes_cli.conversation_history = []
    hermes_cli._secret_capture_callback = lambda *_args, **_kwargs: None

    def _fake_local_governance(instruction_text: str) -> bool:
        captured["instruction_text"] = instruction_text
        hermes_cli.conversation_history.extend([
            {
                "role": "user",
                "content": '{"operator_ingress_surface": "cli_chat"}',
            },
            {
                "role": "assistant",
                "content": "operator_ingress_surface: cli_chat",
            },
        ])
        return True

    monkeypatch.setattr(
        hermes_cli,
        "_run_whatsapp_local_governance_ingress",
        _fake_local_governance,
    )
    monkeypatch.setattr(
        hermes_cli,
        "_ensure_runtime_credentials",
        lambda: (_ for _ in ()).throw(
            AssertionError("runtime credential refresh should be bypassed")
        ),
    )
    monkeypatch.setattr(
        "cli.set_secret_capture_callback",
        lambda *_args, **_kwargs: None,
    )

    result = hermes_cli.chat(
        "whatsapp start approved_destination_chat_id=15551230000@s.whatsapp.net "
        'operator_objective="Request the first quote" '
        'message_text="Hello from Hermes."'
    )

    assert captured["instruction_text"].startswith("whatsapp start ")
    assert result == "operator_ingress_surface: cli_chat"
    assert [entry["role"] for entry in hermes_cli.conversation_history] == [
        "user",
        "assistant",
    ]


def test_cli_chat_short_circuits_handled_governance_turn_without_new_assistant_row(
    monkeypatch,
):
    captured = {}

    hermes_cli = HermesCLI.__new__(HermesCLI)
    hermes_cli.conversation_history = [
        {"role": "assistant", "content": "previous assistant turn"}
    ]
    hermes_cli._secret_capture_callback = lambda *_args, **_kwargs: None

    def _fake_local_governance(instruction_text: str) -> bool:
        captured["instruction_text"] = instruction_text
        return True

    monkeypatch.setattr(
        hermes_cli,
        "_run_whatsapp_local_governance_ingress",
        _fake_local_governance,
    )
    monkeypatch.setattr(
        hermes_cli,
        "_ensure_runtime_credentials",
        lambda: (_ for _ in ()).throw(
            AssertionError("runtime credential refresh should be bypassed")
        ),
    )
    monkeypatch.setattr(
        hermes_cli,
        "_init_agent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("agent init should be bypassed")
        ),
    )
    monkeypatch.setattr(
        "cli.set_secret_capture_callback",
        lambda *_args, **_kwargs: None,
    )

    result = hermes_cli.chat(
        "whatsapp start approved_destination_chat_id=15551230000@s.whatsapp.net "
        'operator_objective="Request the first quote" '
        'message_text="Hello from Hermes."'
    )

    assert captured["instruction_text"].startswith("whatsapp start ")
    assert result == ""
    assert hermes_cli.conversation_history == [
        {"role": "assistant", "content": "previous assistant turn"}
    ]
