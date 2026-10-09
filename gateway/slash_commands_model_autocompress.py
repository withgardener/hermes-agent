"""LOCAL PATCH-022: auto-compress before a large-context ``/model`` switch.

When ``model.switch_context_auto_compress`` is true and the context-cache selection guard fires,
the gateway compresses the session on the CURRENT model (its prompt cache is still warm) and then
switches without asking. Only when compression fails or leaves the transcript above the threshold
does the original confirmation prompt come back.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def auto_compress_enabled() -> bool:
    try:
        from hermes_cli.config import load_config
        from utils import is_truthy_value

        model_cfg = (load_config() or {}).get("model", {})
        raw = model_cfg.get("switch_context_auto_compress") if isinstance(model_cfg, dict) else None
        return is_truthy_value(raw, default=False)
    except Exception:
        return False


def _transcript_tokens(history) -> int:
    from agent.model_metadata import estimate_messages_tokens_rough

    msgs = [m for m in history or [] if m.get("role") in {"user", "assistant", "tool"}]
    return estimate_messages_tokens_rough(msgs) if msgs else 0


async def _send_notice(runner, event, text: str) -> None:
    adapter = runner._delivery_adapter_for(event.source)
    if adapter is None:
        return
    try:
        anchor = runner._reply_anchor_for_event(event)
        await adapter._send_with_retry(
            chat_id=event.source.chat_id, content=text, reply_to=anchor,
            metadata=runner._thread_metadata_for_source(event.source, anchor))
    except Exception as exc:
        logger.debug("auto-compress notice failed: %s", exc)


async def _compress_session(runner, event) -> tuple[bool, str, int, int]:
    """Run manual compression for the event's session. Returns (ran, reply, before, after)."""
    from agent.conversation_compression_manual import MIN_MESSAGES, parse_compress_args

    source = event.source
    session_entry = await runner.async_session_store.get_or_create_session(source)
    history = await runner.async_session_store.load_transcript(session_entry.session_id)
    before = _transcript_tokens(history)
    if not history or len(history) < MIN_MESSAGES:
        return False, "", before, before
    reply = await runner._run_manual_compression(source, session_entry, history, parse_compress_args(""))
    session_entry = await runner.async_session_store.get_or_create_session(source)
    after = _transcript_tokens(await runner.async_session_store.load_transcript(session_entry.session_id))
    return True, reply or "", before, after


async def compress_before_switch(runner, event, warning, threshold: int, live_tokens: int = 0) -> tuple[bool, Optional[str]]:
    """Compress ahead of a switch. Returns (ok_to_switch, report_text). ``live_tokens`` is the billed
    prompt size (system + tools + transcript); its non-transcript overhead is added back to the
    post-compression transcript estimate before comparing against the threshold."""
    await _send_notice(
        runner, event,
        f"🗜️ 上下文较大（{warning.model} 切换前检测），先用当前模型自动压缩再切换，请稍候…")
    try:
        if getattr(getattr(runner, "config", None), "multiplex_profiles", False):
            from gateway.run import _profile_runtime_scope
            with _profile_runtime_scope(runner._resolve_profile_home_for_source(event.source)):
                ran, reply, before, after = await _compress_session(runner, event)
        else:
            ran, reply, before, after = await _compress_session(runner, event)
    except Exception as exc:
        logger.warning("auto-compress before model switch failed: %s", exc)
        return False, f"⚠️ 自动压缩失败：{exc}"
    overhead = max(0, int(live_tokens or 0) - before)
    projected = after + overhead
    report = (reply.strip() + "\n" if reply else "") + (
        f"（切换前自动压缩：预计上下文 ~{before + overhead:,} → ~{projected:,} tokens，阈值 {threshold:,}）")
    if not ran:
        return False, report
    return projected < threshold, report


def split_context_cache(warnings):
    """(context_cache warning or None, remaining warnings)."""
    cache = next((w for w in warnings if w.kind == "context_cache"), None)
    return cache, [w for w in warnings if w.kind != "context_cache"]


async def maybe_auto_compress(runner, event, ctx, result) -> tuple[bool, Optional[str]]:
    """Gateway typed ``/model`` hook. Returns (handled, reply_or_None). ``handled=False`` means the
    caller runs the stock guard flow unchanged."""
    if not auto_compress_enabled():
        return False, None
    from hermes_cli.model_selection_guards import (
        _context_cache_threshold, combined_message, selection_context_for_agent, selection_warnings)

    sel_ctx = selection_context_for_agent(runner._cached_agent_for(ctx.session_key))
    warnings = await asyncio.to_thread(
        selection_warnings, result.new_model, provider=result.target_provider,
        base_url=result.base_url or ctx.current_base_url or "",
        api_key=result.api_key or ctx.current_api_key or "", model_info=result.model_info,
        selection_context=sel_ctx)
    cache_warning, others = split_context_cache(warnings)
    if cache_warning is None:
        return False, None
    if others:
        # Cost / data-policy prompts still need a human; leave the stock flow in charge.
        return False, None
    threshold = _context_cache_threshold()
    ok, report = await compress_before_switch(
        runner, event, cache_warning, threshold, live_tokens=getattr(sel_ctx, "context_tokens", 0) or 0)
    if not ok:
        async def _on_confirm(choice: str) -> str:
            if choice == "cancel":
                return f"🟡 Model switch cancelled. Current model unchanged ({ctx.current_model or 'unknown'})."
            return await runner._commit_model_switch(result, ctx, source=ctx.source)

        p = runner._typed_command_prefix_for(event.source.platform)
        message = (
            f"⚠️ **{cache_warning.title}**\n\n{report}\n\n自动压缩后仍未降到阈值以下。\n\n"
            f"{combined_message([cache_warning])}\n\n"
            f"_Text fallback: reply `{p}approve` to switch or `{p}cancel` to keep the current model._")
        return True, await runner._request_slash_confirm(
            event=event, command="model", title=cache_warning.title, message=message, handler=_on_confirm)
    reply = await runner._commit_model_switch_locked(result, ctx, source=ctx.source, picker=False)
    return True, f"{report}\n\n{reply}"
