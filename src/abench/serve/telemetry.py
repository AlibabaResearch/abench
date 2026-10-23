# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Server-side telemetry: ``/metrics`` scraping and Prometheus parsing.

Backend-specific gauge vocabularies (sglang ``cache_*``/``kv_*``, vLLM
``vllm:*``) are isolated here; the rest of the codebase only ever sees the
normalized ``ServerMetricsSnapshot`` record.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Dict, List, Optional

import aiohttp

from abench.core.metrics import ServerMetricsSnapshot

def _gauge(*names: str, prefix: str = "sglang") -> List[re.Pattern]:
    """Patterns for a gauge under any of ``names``, tolerating labels.

    Newer sglang tags every series with ``{engine_type=...,tp_rank=...}``,
    so the value is not adjacent to the metric name. With tp>1 the same
    gauge is exported once per rank; the ranks agree on these values, so
    the first match is taken.
    """
    return [
        re.compile(
            r'^' + re.escape(prefix) + r':' + re.escape(n)
            + r'(?:\{[^}]*\})?\s+([\d.eE+\-]+)', re.M)
        for n in names
    ]


# Gauge names were renamed between sglang builds (newer builds export
# ``cache_*``, some 0.5.x builds ``kv_*``), so each field lists its
# candidates newest-first.
_METRIC_PATTERNS: Dict[str, List[re.Pattern]] = {
    "cache_available_tokens": _gauge("kv_available_tokens", "cache_available_tokens"),
    "cache_used_tokens": _gauge("kv_used_tokens", "cache_used_tokens"),
    "cache_evictable_tokens": _gauge("kv_evictable_tokens", "cache_evictable_tokens"),
    # Tokens on newer builds, requests on older ones. Only ever read as a
    # delta ("did the run evict anything"), so either unit answers that.
    "num_evicted_requests": _gauge("evicted_tokens_total", "num_evicted_requests_total"),
    "num_running_reqs": _gauge("num_running_reqs"),
    "num_waiting_reqs": _gauge("num_queue_reqs", "num_waiting_reqs"),
    # Utilization and capacity straight from the server, independent of
    # whether the three pool gauges above resolve.
    "token_usage": _gauge("token_usage"),
    "max_total_tokens": _gauge("max_total_num_tokens"),
    # HiCache / unified-cache diagnostics (host L2, eviction)
    "cache_hit_rate": _gauge("cache_hit_rate"),
    "evicted_tokens_total": _gauge("evicted_tokens_total"),
    "kv_evictable_tokens": _gauge("kv_evictable_tokens"),
    "mamba_evictable_tokens": _gauge("mamba_evictable_tokens"),
    "hicache_host_used_tokens": _gauge("hicache_host_used_tokens"),
    "hicache_host_total_tokens": _gauge("hicache_host_total_tokens"),
}


# vLLM /metrics gauges/counters mapped onto the snapshot fields they can
# fill (verified against vLLM v0.10.x source; names drift across versions).
# num_preemptions is the v1 cumulative preemption counter (eviction
# equivalent); num_requests_preempted was the v0 gauge. Both are only ever
# read as a delta. token_usage: vllm:kv_cache_usage_perc supersedes
# gpu_cache_usage_perc on newer builds. Everything else stays 0 (reported
# as null downstream); KV pool capacity comes from the cache_config_info
# info-gauge labels, parsed separately below.
_VLLM_METRIC_PATTERNS: Dict[str, List[re.Pattern]] = {
    "num_running_reqs": _gauge("num_requests_running", prefix="vllm"),
    "num_waiting_reqs": _gauge("num_requests_waiting", prefix="vllm"),
    "token_usage": _gauge("gpu_cache_usage_perc", "kv_cache_usage_perc", prefix="vllm"),
    "num_evicted_requests": _gauge(
        "num_preemptions", "num_requests_preempted", prefix="vllm"),
    # Cumulative counters; the run's prefix-cache ratio comes from their
    # deltas (usage.prompt_tokens_details is null on current vLLM builds).
    "prefix_cache_queries_total": _gauge(
        "prefix_cache_queries_total", prefix="vllm"),
    "prefix_cache_hits_total": _gauge(
        "prefix_cache_hits_total", prefix="vllm"),
}


def _parse_vllm_cache_config(text: str) -> float:
    """KV pool capacity in tokens from the cache_config_info info-gauge.

    vLLM exports its CacheConfig as a gauge permanently set to 1 whose
    labels carry the config dict; capacity = num_gpu_blocks * block_size.
    num_gpu_blocks is only set once the engine has initialized (it is the
    string "None" before), in which case the series is skipped.
    """
    for m in re.finditer(r'^vllm:cache_config_info\{([^}]*)\}', text, re.M):
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        try:
            blocks = float(labels.get("num_gpu_blocks", "nan"))
            block_size = float(labels.get("block_size", "nan"))
        except ValueError:
            continue
        if blocks == blocks and block_size == block_size:  # not NaN
            return blocks * block_size
    return 0.0


def _parse_prometheus_text(text: str, backend: str = "sglang") -> ServerMetricsSnapshot:
    """Extract known gauges/counters from raw Prometheus exposition text."""
    snap = ServerMetricsSnapshot(timestamp=time.perf_counter())
    patterns = _VLLM_METRIC_PATTERNS if backend == "vllm" else _METRIC_PATTERNS
    for attr, pats in patterns.items():
        for pat in pats:
            m = pat.search(text)
            if m:
                setattr(snap, attr, float(m.group(1)))
                break
    if backend == "vllm":
        snap.max_total_tokens = _parse_vllm_cache_config(text)
    else:
        # cached_tokens_total carries a cache_source label
        # and is exported once per source; sum every line.
        snap.cached_tokens_total = sum(
            float(v) for v in re.findall(
                r'^sglang:cached_tokens_total(?:\{[^}]*\})?\s+([\d.eE+\-]+)', text, re.M))
    return snap


async def _scrape_server_metrics(
    http: aiohttp.ClientSession,
    metrics_url: str,
    interval: float,
    snapshots: List[ServerMetricsSnapshot],
    verbose: bool = False,
    backend: str = "sglang",
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """Periodically GET /metrics; silently disables itself on first failure."""
    try:
        async with http.get(metrics_url, headers=headers or {}) as resp:
            if resp.status != 200:
                print(
                    f"[metrics] /metrics returned {resp.status}; "
                    "server-side KV stats skipped",
                    flush=True,
                )
                return
            text = await resp.text()
            snapshots.append(_parse_prometheus_text(text, backend))
    except Exception:
        print(
            "[metrics] /metrics unreachable; server-side KV stats skipped",
            flush=True,
        )
        return
    if verbose:
        print(f"[metrics] scraping {metrics_url} every {interval:.0f}s", flush=True)
    while True:
        await asyncio.sleep(interval)
        try:
            async with http.get(metrics_url, headers=headers or {}) as resp:
                if resp.status == 200:
                    snapshots.append(
                        _parse_prometheus_text(await resp.text(), backend)
                    )
        except Exception:
            pass  # transient failure; keep trying
