"""PATCH-032: the summary call states its own output cap on OpenAI-compatible gateways.

Those gateways replace the model's output ceiling with their own default when a request omits
``max_tokens`` (the okhema gateway stops Claude at exactly 8,192 completion tokens), so a summary
whose prompt target exceeds it is cut mid-sentence and rejected as ``finish_reason=length``.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

import agent.context_compressor as cc
from agent.auxiliary_client import _build_call_kwargs, _forwards_max_tokens

GATEWAY = "https://n1-1.okhema.cn/v1"
MODEL = "CLAUDE/Sonnet-5.5"


class _Usage:
    prompt_tokens = 10
    completion_tokens = 10
    total_tokens = 20


class _Message:
    reasoning_content = None

    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = _Message(content)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content: str = "## Summary\nok", finish_reason: str = "stop") -> None:
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage()


def _compressor(**overrides: Any) -> cc.ContextCompressor:
    kwargs: dict[str, Any] = dict(
        model=MODEL, provider="custom:OKHEMA", base_url=GATEWAY, api_key="sk-test",
        max_tokens=64000, config_context_length=1_000_000, quiet_mode=True,
    )
    kwargs.update(overrides)
    return cc.ContextCompressor(**kwargs)


def _record_summary_call(monkeypatch, compressor: cc.ContextCompressor, budget: int) -> dict[str, Any]:
    """Run _call_summary_llm with a recording stub; return the kwargs it handed to call_llm."""
    seen: list[dict[str, Any]] = []

    def _fake_call(**kwargs):
        seen.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(cc, "call_llm", _fake_call)
    compressor._call_summary_llm("summarize this", time.monotonic(), budget)
    assert seen, "the summary call never reached call_llm"
    return seen[0]


def test_summary_call_states_cap_on_custom_gateway(monkeypatch):
    """A custom OpenAI-compatible gateway gets an explicit cap above the prompt's own target."""
    kwargs = _record_summary_call(monkeypatch, _compressor(), 9229)

    assert kwargs["task"] == "compression"
    # 2 x the summary budget leaves room for the target plus structure; the gateway's 8K default
    # would have cut this at 8,192.
    assert kwargs["max_tokens"] == 2 * 9229


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "nous", ""])
def test_summary_call_omits_cap_on_other_providers(monkeypatch, provider):
    """Every other route keeps omitting max_tokens (the provider's own maximum applies)."""
    kwargs = _record_summary_call(monkeypatch, _compressor(provider=provider, base_url="", api_key=""), 9229)

    assert "max_tokens" not in kwargs


def test_wire_cap_floor_ceiling_and_model_limit():
    """Floor 16K, ceiling 64K, clamped by the main model's own output limit."""
    small = _compressor()
    assert small._summary_wire_max_tokens(500) == 16_000

    big = _compressor()
    assert big._summary_wire_max_tokens(40_000) == 64_000

    limited = _compressor(max_tokens=8192)
    assert limited._summary_wire_max_tokens(40_000) == 8192

    unlimited = _compressor(max_tokens=None)
    assert unlimited._summary_wire_max_tokens(40_000) == 64_000

    other = _compressor(provider="anthropic")
    assert other._summary_wire_max_tokens(40_000) is None


def test_custom_gateway_forwards_explicit_compression_cap():
    """The cap survives _build_call_kwargs on a custom gateway instead of being dropped."""
    kwargs = _build_call_kwargs(
        "custom:OKHEMA", MODEL, [{"role": "user", "content": "x"}],
        max_tokens=18_458, base_url=GATEWAY, task="compression",
    )
    assert kwargs["max_tokens"] == 18_458
    assert "max_completion_tokens" not in kwargs


def test_other_tasks_and_providers_still_drop_the_cap():
    """The forwarding exception is scoped to compression on a custom gateway."""
    non_compression = _build_call_kwargs(
        "custom:OKHEMA", MODEL, [{"role": "user", "content": "x"}],
        max_tokens=18_458, base_url=GATEWAY, task="title",
    )
    assert "max_tokens" not in non_compression

    other_provider = _build_call_kwargs(
        "openai", MODEL, [{"role": "user", "content": "x"}],
        max_tokens=18_458, base_url="https://api.openai.com/v1", task="compression",
    )
    assert "max_tokens" not in other_provider


def test_custom_gateway_helper_is_narrow():
    from agent.auxiliary_client import _is_custom_openai_gateway

    assert _is_custom_openai_gateway("custom") is True
    assert _is_custom_openai_gateway("custom:OKHEMA") is True
    assert _is_custom_openai_gateway("customary") is False
    assert _is_custom_openai_gateway(None) is False


def test_truncated_summary_is_still_a_failure(monkeypatch):
    """Stating the cap must not weaken the guard: a length stop is still rejected."""
    compressor = _compressor()

    def _truncated(**kwargs):
        return FakeResponse("partial", finish_reason="length")

    monkeypatch.setattr(cc, "call_llm", _truncated)
    with pytest.raises(RuntimeError, match="truncated"):
        compressor._call_summary_llm("summarize this", time.monotonic(), 9229)


def test_forwards_max_tokens_scopes_the_exception():
    assert _forwards_max_tokens("custom:OKHEMA", "okhema", MODEL, GATEWAY, "compression") is True
    assert _forwards_max_tokens("custom", "custom", MODEL, GATEWAY, "compression") is True
    assert _forwards_max_tokens("custom:OKHEMA", "okhema", MODEL, GATEWAY, "title") is False
    # Pre-existing exceptions stay intact.
    assert _forwards_max_tokens("openrouter", "openrouter", MODEL, "https://openrouter.ai/api/v1", "compression") is True
