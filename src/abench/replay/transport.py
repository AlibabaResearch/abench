# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Turn-level request/response plumbing for the replay engine.

Token counting and the streaming / non-streaming response processors: the
parts that turn one HTTP exchange into one measured ``TurnRecord`` input.
Scheduling and concurrency live in ``abench.replay.engine``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# ----------------------------------------------------------------------
# Tokenizer helpers
# ----------------------------------------------------------------------
def count_tokens(text: str, tokenizer: Optional[Any]) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


# ----------------------------------------------------------------------
# SSE / response processing
# ----------------------------------------------------------------------
def _read_cache_details(obj: Dict[str, Any], usage_out: Dict[str, int]) -> None:
    """Read the per-layer prefix-cache breakdown into ``usage_out``.

    sglang reports it outside ``usage``, as a top-level
    ``sglext.cached_tokens_details`` of ``{device, host}``. Absent unless the
    request set ``return_cached_tokens_details``.
    """
    details = (obj.get("sglext") or {}).get("cached_tokens_details")
    if not details:
        return
    usage_out["cached_device_tokens"] = int(details.get("device", 0) or 0)
    usage_out["cached_host_tokens"] = int(details.get("host", 0) or 0)


async def _process_stream(
    response: aiohttp.ClientResponse, start_time: float
) -> Tuple[str, float, List[float], Dict[str, int]]:
    """Consume an OpenAI-style SSE stream.

    Returns ``(text, ttft_s, itl_list, usage)`` where ``usage`` carries the
    server-reported ``prompt_tokens`` / ``completion_tokens`` /
    ``cached_tokens`` (0 when the server does not include them).
    """
    text = ""
    ttft = 0.0
    itl: List[float] = []
    last = start_time
    usage_out = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cached_device_tokens": 0,
        "cached_host_tokens": 0,
    }

    async for chunk in response.content:
        try:
            s = chunk.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line in s.split("\n"):
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            _read_cache_details(data, usage_out)
            usage = data.get("usage")
            if usage:
                usage_out["prompt_tokens"] = int(
                    usage.get("prompt_tokens", 0) or 0
                )
                usage_out["completion_tokens"] = int(
                    usage.get("completion_tokens", 0) or 0
                )
                details = usage.get("prompt_tokens_details") or {}
                usage_out["cached_tokens"] = int(
                    details.get("cached_tokens", 0) or 0
                )
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}) or {}
            content = (delta.get("reasoning_content") or "") + (
                delta.get("content") or ""
            )
            if content:
                now = time.perf_counter()
                if ttft == 0.0:
                    ttft = now - start_time
                else:
                    itl.append(now - last)
                last = now
                text += content
    return text, ttft, itl, usage_out


async def _process_non_stream(
    response: aiohttp.ClientResponse, start_time: float
) -> Tuple[str, float, List[float], Dict[str, int]]:
    payload = await response.json()
    elapsed = time.perf_counter() - start_time
    text = ""
    usage_out = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cached_device_tokens": 0,
        "cached_host_tokens": 0,
    }
    if isinstance(payload, dict):
        _read_cache_details(payload, usage_out)
        usage = payload.get("usage") or {}
        usage_out["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
        usage_out["completion_tokens"] = int(
            usage.get("completion_tokens", 0) or 0
        )
        details = usage.get("prompt_tokens_details") or {}
        usage_out["cached_tokens"] = int(details.get("cached_tokens", 0) or 0)
        choices = payload.get("choices") or []
        if choices:
            msg = choices[0].get("message", {}) or {}
            text = (msg.get("reasoning_content") or "") + (
                msg.get("content") or ""
            )
    return text, elapsed, [], usage_out


async def _count_tokens_async(text: str, tokenizer: Optional[Any]) -> int:
    """Tokenize off the event loop so in-flight latency samples stay clean."""
    if not text:
        return 0
    if tokenizer is None:
        return max(1, len(text) // 4)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, count_tokens, text, tokenizer)
