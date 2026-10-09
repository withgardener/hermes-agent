"""LOCAL PATCH-022: ``model.switch_context_auto_compress`` compresses before a large-context switch."""

from types import SimpleNamespace

import pytest

import gateway.slash_commands_model_autocompress as ac
from hermes_cli.model_selection_guards import SelectionWarning


def _w(kind):
    return SelectionWarning(kind=kind, title=f"{kind} title", model="new/model", provider="p", message=f"{kind} msg")


class _Runner:
    def __init__(self):
        self.committed = []
        self.confirm = None
        self.config = SimpleNamespace(multiplex_profiles=False)

    def _cached_agent_for(self, _key):
        return SimpleNamespace(context_compressor=SimpleNamespace(last_prompt_tokens=150_000), model="old/model")

    def _delivery_adapter_for(self, _source):
        return None

    def _typed_command_prefix_for(self, _platform):
        return "/"

    async def _commit_model_switch_locked(self, result, ctx, *, source, picker):
        self.committed.append(result.new_model)
        return f"switched to {result.new_model}"

    async def _commit_model_switch(self, result, ctx, *, source, picker=False):
        return await self._commit_model_switch_locked(result, ctx, source=source, picker=picker)

    async def _request_slash_confirm(self, **kw):
        self.confirm = kw
        return None


def _args():
    event = SimpleNamespace(source=SimpleNamespace(platform="feishu", chat_id="c"))
    ctx = SimpleNamespace(session_key="k", source=event.source, current_base_url="", current_api_key="",
                          current_model="old/model")
    result = SimpleNamespace(new_model="new/model", target_provider="p", base_url="", api_key="", model_info=None)
    return event, ctx, result


def _patch(monkeypatch, *, enabled, warnings, compress):
    monkeypatch.setattr(ac, "auto_compress_enabled", lambda: enabled)
    monkeypatch.setattr("hermes_cli.model_selection_guards.selection_warnings", lambda *a, **k: warnings)
    monkeypatch.setattr("hermes_cli.model_selection_guards._context_cache_threshold", lambda: 100_000)

    async def _fake_compress(runner, event):
        return compress

    monkeypatch.setattr(ac, "_compress_session", _fake_compress)


@pytest.mark.asyncio
async def test_disabled_leaves_stock_flow(monkeypatch):
    _patch(monkeypatch, enabled=False, warnings=[_w("context_cache")], compress=(True, "", 140_000, 20_000))
    runner = _Runner()
    assert await ac.maybe_auto_compress(runner, *_args()) == (False, None)
    assert runner.committed == []


@pytest.mark.asyncio
async def test_compress_below_threshold_switches_without_prompt(monkeypatch):
    _patch(monkeypatch, enabled=True, warnings=[_w("context_cache")], compress=(True, "🗜️ done", 140_000, 20_000))
    runner = _Runner()
    handled, reply = await ac.maybe_auto_compress(runner, *_args())
    assert handled and runner.committed == ["new/model"] and runner.confirm is None
    assert "switched to new/model" in reply and "🗜️ done" in reply


@pytest.mark.asyncio
async def test_still_above_threshold_falls_back_to_confirm(monkeypatch):
    # live 150k, transcript 140k -> 10k overhead; 95k transcript after + 10k = 105k >= 100k
    _patch(monkeypatch, enabled=True, warnings=[_w("context_cache")], compress=(True, "", 140_000, 95_000))
    runner = _Runner()
    handled, reply = await ac.maybe_auto_compress(runner, *_args())
    assert handled and reply is None and runner.committed == []
    assert runner.confirm is not None
    assert await runner.confirm["handler"]("cancel") and runner.committed == []
    await runner.confirm["handler"]("once")
    assert runner.committed == ["new/model"]


@pytest.mark.asyncio
async def test_other_guards_keep_stock_flow(monkeypatch):
    _patch(monkeypatch, enabled=True, warnings=[_w("context_cache"), _w("cost")], compress=(True, "", 1, 1))
    runner = _Runner()
    assert await ac.maybe_auto_compress(runner, *_args()) == (False, None)


@pytest.mark.asyncio
async def test_no_cache_warning_is_noop(monkeypatch):
    _patch(monkeypatch, enabled=True, warnings=[], compress=(True, "", 1, 1))
    assert await ac.maybe_auto_compress(_Runner(), *_args()) == (False, None)
