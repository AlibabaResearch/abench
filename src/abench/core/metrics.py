# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Metrics for closed-loop agentic replay.

The central design choices here address the two fidelity problems that
plague fixed-concurrency agent benchmarks:

* **Long-tail / ramp-drain bias** — at the start (ramp) and end (drain)
  of a closed-loop run the system is under-loaded, so turns completing in
  those windows see artificially low latency. We tag every turn with the
  number of *active sessions* at dispatch time and, by default, compute
  headline metrics only over the **steady-state window** where the active
  count is at (or near) the target concurrency ``C``. The achieved
  concurrency distribution is always reported so the trim is auditable.

* **Heterogeneous turn shape** — a turn-1 request (tiny context) and a
  turn-30 request (huge context) are different beasts; averaging them
  hides everything. All latency metrics are therefore **bucketed by turn
  index** (T1 vs. T2+) and, optionally, by input-length band.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TurnRecord:
    """Everything measured for a single issued turn."""

    session_id: str
    turn_index: int
    input_len: int
    output_len: int = 0
    ttft: float = 0.0            # seconds
    latency: float = 0.0         # seconds (server E2E for the turn)
    itl: List[float] = field(default_factory=list)
    server_cached_tokens: int = 0
    # Breakdown of ``server_cached_tokens`` by cache layer. The total already
    # includes host hits, so only the split shows whether the DRAM layer (L2)
    # is doing any work. Requires ``return_cached_tokens_details``.
    cached_device_tokens: int = 0
    cached_host_tokens: int = 0
    recorded_output_len: int = 0  # what the trace generated for this turn
    dispatch_time: float = 0.0   # perf_counter at request send
    complete_time: float = 0.0   # perf_counter at response end
    dispatch_epoch: float = 0.0  # wall-clock epoch (s) at request send
    complete_epoch: float = 0.0  # wall-clock epoch (s) at response end
    active_at_dispatch: int = 0
    success: bool = False
    error: str = ""


@dataclass
class SessionRecord:
    """Per-session roll-up (trajectory inference time, worker)."""

    session_id: str
    source_model: str = ""  # model that produced the recorded trajectory
    turns: List[TurnRecord] = field(default_factory=list)
    assigned_workers: List[str] = field(default_factory=list)
    # True when the benchmark stopped this session early because the pool
    # was exhausted (stop-on-exhaustion). Not a failure: such sessions are
    # excluded from session-level completion/trajectory statistics.
    truncated: bool = False
    # Subset of ``truncated``: the session was stopped by the client-side
    # context guard because the estimated prompt + pinned completion no
    # longer fit the model's context window.
    ctx_guard_stop: bool = False

    @property
    def trajectory_inference_time(self) -> float:
        return sum(t.latency for t in self.turns if t.success)


# ----------------------------------------------------------------------
# Small statistics helpers
# ----------------------------------------------------------------------
def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class BucketStats:
    label: str
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float


def _summarize(label: str, secs: List[float]) -> BucketStats:
    ms = [x * 1000.0 for x in secs]
    return BucketStats(
        label=label,
        count=len(ms),
        mean_ms=_mean(ms),
        p50_ms=_pct(ms, 0.50),
        p90_ms=_pct(ms, 0.90),
        p95_ms=_pct(ms, 0.95),
        p99_ms=_pct(ms, 0.99),
    )


# ----------------------------------------------------------------------
# Aggregate metrics
# ----------------------------------------------------------------------
@dataclass
class AgenticMetrics:
    # window / load audit
    target_concurrency: int = 0
    steady_frac: float = 0.9
    total_turns: int = 0
    steady_turns: int = 0
    concurrency_hist: Dict[int, int] = field(default_factory=dict)
    steady_window_s: float = 0.0

    # Whole-run roll-up (bench_serving-style header). Unlike the steady
    # window above, these cover every successful turn of the run: wall
    # clock is first dispatch -> last completion (wall-clock epochs), so
    # wall_duration_s >= steady_window_s by construction.
    successful_turns: int = 0
    wall_duration_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turns_throughput: float = 0.0
    input_throughput: float = 0.0
    # Peak per-second output rate: each turn's output tokens are spread
    # evenly across the 1s buckets it spans, then the busiest bucket wins.
    peak_output_throughput: float = 0.0
    e2e_overall: Optional[BucketStats] = None
    concurrency_mean: float = 0.0
    concurrency_peak: int = 0

    # latency buckets (steady window)
    t1_ttft: Optional[BucketStats] = None
    t2plus_ttft: Optional[BucketStats] = None
    tpot_ms: float = 0.0
    # Inter-token latency distribution over the steady window. The mean
    # (tpot_ms) hides mid-decode stalls — chunked-prefill preemption and
    # KV-eviction recompute both show up as ITL tail, not as mean.
    itl_p50_ms: float = 0.0
    itl_p90_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0
    itl_max_ms: float = 0.0
    e2e_by_turn: List[BucketStats] = field(default_factory=list)
    e2e_by_inputlen: List[BucketStats] = field(default_factory=list)
    ttft_degradation: float = 0.0

    # cache / throughput
    prefix_cache_ratio: float = 0.0
    # Host-layer cache split. Only SGLang reports it (sglext); ``None``
    # when the backend cannot measure it (not measured != zero).
    host_cache_ratio: Optional[float] = 0.0
    host_share_of_cache: Optional[float] = 0.0
    output_throughput: float = 0.0
    total_throughput: float = 0.0
    # Pareto point, as used by MLPerf's agentic methodology: the system axis
    # is aggregate output tokens/s, the per-user axis divides a trajectory's
    # output tokens by its serving time only, with injected tool waits
    # excluded rather than charged to the server.
    per_user_output_tps: float = 0.0
    # Achieved vs recorded mean output length. Lengths are pinned per turn,
    # so this is a fidelity check on the server: a mean outside the band
    # means the engine did not honor the pin.
    osl_mean_achieved: float = 0.0
    osl_mean_expected: float = 0.0
    osl_tolerance: float = 0.10
    osl_within_band: bool = False

    # session level
    num_sessions: int = 0
    completed_sessions: int = 0
    truncated_sessions: int = 0
    ctx_guard_sessions: int = 0
    mean_trajectory_time_s: float = 0.0
    completion_rate: float = 0.0

    # Fraction of sessions whose turns all hit one worker. ``None`` when the
    # run had no worker header at all (single endpoint, no router), where a
    # stickiness number would be meaningless rather than zero.
    routing_stickiness: Optional[float] = None

    # server-side KV pool (from Prometheus /metrics scraping).
    # capacity/eviction come from SGLang-only gauges: ``None`` when the
    # backend does not expose them (not measurable != zero).
    kv_pool_capacity_tokens: Optional[float] = None  # avail+used+evictable (median)
    kv_pool_utilization_mean: Optional[float] = None  # used / capacity, mean
    kv_pool_utilization_p99: Optional[float] = None
    kv_eviction_delta: Optional[float] = None  # total evictions during the run
    kv_queue_depth_p99: Optional[float] = None  # num_waiting_reqs p99
    kv_running_mean: Optional[float] = None  # num_running_reqs mean
    kv_snapshots_count: int = 0  # how many scrapes succeeded


_INPUT_LEN_BUCKETS: List[Tuple[int, int, str]] = [
    (0, 2_000, "0-2K"),
    (2_000, 8_000, "2-8K"),
    (8_000, 16_000, "8-16K"),
    (16_000, 32_000, "16-32K"),
    (32_000, 1_000_000_000, "32K+"),
]


def _inputlen_label(n: int) -> str:
    for lo, hi, lbl in _INPUT_LEN_BUCKETS:
        if lo <= n < hi:
            return lbl
    return "32K+"


def compute_metrics(
    sessions: List[SessionRecord],
    target_concurrency: int,
    steady_frac: float = 0.9,
    steady_only: bool = True,
) -> AgenticMetrics:
    """Aggregate raw records into headline metrics.

    Args:
        sessions: per-session records collected by the replay engine.
        target_concurrency: the configured ``C``.
        steady_frac: a turn is "steady" if ``active_at_dispatch >=
            ceil(steady_frac * C)``. Filters ramp/drain.
        steady_only: when False, all turns are used (no trim); useful for
            debugging or open-loop-style runs.
    """
    all_turns: List[TurnRecord] = [t for s in sessions for t in s.turns]
    m = AgenticMetrics(
        target_concurrency=target_concurrency,
        steady_frac=steady_frac,
        total_turns=len(all_turns),
        num_sessions=len(sessions),
    )

    # concurrency audit (over successful turns)
    m.concurrency_hist = dict(
        Counter(t.active_at_dispatch for t in all_turns if t.success)
    )
    if m.concurrency_hist:
        hist_total = sum(m.concurrency_hist.values())
        m.concurrency_mean = (
            sum(c * n for c, n in m.concurrency_hist.items()) / hist_total
        )
        m.concurrency_peak = max(m.concurrency_hist)

    # Whole-run roll-up over successful turns (bench_serving-style header).
    ok = [t for t in all_turns if t.success and t.complete_epoch > 0]
    m.successful_turns = len(ok)
    m.e2e_overall = _summarize("overall", [t.latency for t in ok])
    if ok:
        w0 = min(t.dispatch_epoch for t in ok)
        w1 = max(t.complete_epoch for t in ok)
        m.wall_duration_s = max(1e-6, w1 - w0)
        m.total_input_tokens = sum(t.input_len for t in ok)
        m.total_output_tokens = sum(t.output_len for t in ok)
        m.turns_throughput = len(ok) / m.wall_duration_s
        m.input_throughput = m.total_input_tokens / m.wall_duration_s
        buckets: Dict[int, float] = defaultdict(float)
        for t in ok:
            b0 = int(t.dispatch_epoch)
            b1 = int(t.complete_epoch)
            per = t.output_len / max(1, b1 - b0 + 1)
            for b in range(b0, b1 + 1):
                buckets[b] += per
        m.peak_output_throughput = max(buckets.values()) if buckets else 0.0

    threshold = math.ceil(steady_frac * target_concurrency) if target_concurrency else 0
    def _in_window(t: TurnRecord) -> bool:
        if not t.success:
            return False
        if not steady_only:
            return True
        return t.active_at_dispatch >= threshold

    win = [t for t in all_turns if _in_window(t)]
    m.steady_turns = len(win)

    # latency buckets
    t1 = [t.ttft for t in win if t.turn_index == 0]
    t2 = [t.ttft for t in win if t.turn_index >= 1]
    m.t1_ttft = _summarize("T1", t1)
    m.t2plus_ttft = _summarize("T2+", t2)
    if m.t1_ttft.mean_ms > 0:
        m.ttft_degradation = m.t2plus_ttft.mean_ms / m.t1_ttft.mean_ms

    all_itl = [x for t in win for x in t.itl]
    itl_ms = [x * 1000.0 for x in all_itl]
    m.tpot_ms = _mean(itl_ms)
    m.itl_p50_ms = _pct(itl_ms, 0.50)
    m.itl_p90_ms = _pct(itl_ms, 0.90)
    m.itl_p95_ms = _pct(itl_ms, 0.95)
    m.itl_p99_ms = _pct(itl_ms, 0.99)
    m.itl_max_ms = max(itl_ms) if itl_ms else 0.0

    # E2E bucketed by turn index (cap the number of explicit buckets)
    by_turn: Dict[int, List[float]] = defaultdict(list)
    for t in win:
        key = t.turn_index if t.turn_index < 10 else 10
        by_turn[key].append(t.latency)
    for k in sorted(by_turn):
        lbl = f"turn>={k}" if k == 10 else f"turn{k}"
        m.e2e_by_turn.append(_summarize(lbl, by_turn[k]))

    # E2E bucketed by input length
    by_len: Dict[str, List[float]] = defaultdict(list)
    for t in win:
        by_len[_inputlen_label(t.input_len)].append(t.latency)
    for _, _, lbl in _INPUT_LEN_BUCKETS:
        if lbl in by_len:
            m.e2e_by_inputlen.append(_summarize(lbl, by_len[lbl]))

    # Prefix cache ratio: token-weighted (sum cached / sum input), so long
    # turns aren't given the same weight as short ones. Turns where the
    # server reported no usage are excluded rather than counted as 0%.
    cached_sum = sum(t.server_cached_tokens for t in win if t.input_len > 0)
    input_sum = sum(t.input_len for t in win if t.input_len > 0)
    m.prefix_cache_ratio = min(1.0, cached_sum / input_sum) if input_sum else 0.0

    # Where the hits came from. ``prefix_cache_ratio`` already counts host
    # hits, so only this split distinguishes an HBM hit from a DRAM one.
    host_sum = sum(t.cached_host_tokens for t in win if t.input_len > 0)
    m.host_cache_ratio = min(1.0, host_sum / input_sum) if input_sum else 0.0
    m.host_share_of_cache = min(1.0, host_sum / cached_sum) if cached_sum else 0.0

    # throughput over the steady wall-clock window
    if win:
        w0 = min(t.dispatch_time for t in win)
        w1 = max(t.complete_time for t in win)
        wall = max(1e-6, w1 - w0)
        m.steady_window_s = wall
        out_tok = sum(t.output_len for t in win)
        in_tok = sum(t.input_len for t in win)
        m.output_throughput = out_tok / wall
        m.total_throughput = (out_tok + in_tok) / wall

    # Session level. Sessions the benchmark itself truncated (pool
    # exhaustion) are neither "completed" nor "failed" — they are reported
    # separately and excluded from trajectory statistics.
    intact = [s for s in sessions if not s.truncated]
    m.truncated_sessions = len(sessions) - len(intact)
    m.ctx_guard_sessions = sum(1 for s in sessions if s.ctx_guard_stop)
    completed = [s for s in intact if s.turns and all(t.success for t in s.turns)]
    m.completed_sessions = len(completed)
    intact_turns = [t for s in intact for t in s.turns]
    m.completion_rate = (
        (sum(1 for t in intact_turns if t.success) / len(intact_turns))
        if intact_turns
        else 0.0
    )
    m.mean_trajectory_time_s = _mean(
        [s.trajectory_inference_time for s in completed]
    )

    # Per-user progression speed. ``trajectory_inference_time`` sums per-turn
    # server time, so the tool waits slept between turns are already out.
    per_user = [
        sum(t.output_len for t in s.turns) / s.trajectory_inference_time
        for s in completed
        if s.trajectory_inference_time > 0
    ]
    m.per_user_output_tps = _mean(per_user)

    # Output-length fidelity over the measured window.
    if win:
        m.osl_mean_achieved = _mean([float(t.output_len) for t in win])
        expected = [float(t.recorded_output_len) for t in win if t.recorded_output_len]
        m.osl_mean_expected = _mean(expected)
        if m.osl_mean_expected > 0:
            lo = m.osl_mean_expected * (1.0 - m.osl_tolerance)
            hi = m.osl_mean_expected * (1.0 + m.osl_tolerance)
            m.osl_within_band = lo <= m.osl_mean_achieved <= hi

    # routing stickiness: fraction of sessions whose turns all hit one worker
    sticky_count = 0
    sessions_with_workers = 0
    for s in sessions:
        if s.assigned_workers:
            sessions_with_workers += 1
            if len(set(s.assigned_workers)) == 1:
                sticky_count += 1
    m.routing_stickiness = (
        sticky_count / sessions_with_workers if sessions_with_workers else None
    )

    return m


def summarize_server_metrics(
    m: AgenticMetrics, snapshots: list, backend: str = "sglang"
) -> None:
    """Populate KV pool fields on ``m`` from raw Prometheus snapshots.

    Called after ``compute_metrics``; a no-op when ``snapshots`` is empty
    (server did not expose /metrics). Metrics the backend cannot provide
    stay ``None`` rather than being reported as zero.
    """
    if not snapshots:
        return
    m.kv_snapshots_count = len(snapshots)

    waiting = [s.num_waiting_reqs for s in snapshots]
    m.kv_queue_depth_p99 = _pct(waiting, 0.99)

    running = [s.num_running_reqs for s in snapshots]
    m.kv_running_mean = _mean(running)

    if backend == "vllm":
        # vLLM: the cache usage gauge is already used/capacity; capacity
        # comes from the cache_config_info info-gauge labels and eviction
        # from the num_preemptions counter (both parsed at scrape time).
        utils = [s.token_usage for s in snapshots]
        m.kv_pool_utilization_mean = _mean(utils)
        m.kv_pool_utilization_p99 = _pct(utils, 0.99)
        caps = [s.max_total_tokens for s in snapshots if s.max_total_tokens > 0]
        if caps:
            m.kv_pool_capacity_tokens = _pct(caps, 0.5)
        m.kv_eviction_delta = max(
            0.0, snapshots[-1].num_evicted_requests - snapshots[0].num_evicted_requests
        )
        # Client-side cached_tokens is null on current vLLM builds; the
        # prefix-cache ratio comes from the server counters instead.
        dq = max(0.0, snapshots[-1].prefix_cache_queries_total - snapshots[0].prefix_cache_queries_total)
        dh = max(0.0, snapshots[-1].prefix_cache_hits_total - snapshots[0].prefix_cache_hits_total)
        if dq > 0:
            m.prefix_cache_ratio = min(1.0, dh / dq)
        return

    capacities = [
        s.cache_available_tokens + s.cache_used_tokens + s.cache_evictable_tokens
        for s in snapshots
    ]
    # The three pool gauges are absent on some builds; max_total_num_tokens
    # and token_usage carry the same information and are always present.
    if max(capacities, default=0.0) <= 0:
        capacities = [s.max_total_tokens for s in snapshots]
        utils = [s.token_usage for s in snapshots]
    else:
        utils = [
            s.cache_used_tokens / cap if cap > 0 else 0.0
            for s, cap in zip(snapshots, capacities)
        ]
    m.kv_pool_capacity_tokens = _pct(capacities, 0.5)
    m.kv_pool_utilization_mean = _mean(utils)
    m.kv_pool_utilization_p99 = _pct(utils, 0.99)

    # Eviction delta: last counter value minus first.
    m.kv_eviction_delta = max(
        0.0, snapshots[-1].num_evicted_requests - snapshots[0].num_evicted_requests
    )


# ----------------------------------------------------------------------
# Server-side Prometheus metrics scraping
# ----------------------------------------------------------------------
@dataclass
class ServerMetricsSnapshot:
    """One scrape of the server's /metrics, normalized across engines.

    Filled by abench.serve.telemetry; kept here with the other result
    records so the core layer stays free of HTTP concerns.
    """

    timestamp: float  # perf_counter at scrape time
    cache_available_tokens: float = 0.0
    cache_used_tokens: float = 0.0
    cache_evictable_tokens: float = 0.0
    num_evicted_requests: float = 0.0  # COUNTER (cumulative)
    num_running_reqs: float = 0.0
    num_waiting_reqs: float = 0.0
    token_usage: float = 0.0  # used / capacity, as the server computes it
    max_total_tokens: float = 0.0  # KV pool capacity
    # Unified cache / HiCache host-layer metrics
    cache_hit_rate: float = 0.0
    cached_tokens_total: float = 0.0  # COUNTER, device+host summed
    evicted_tokens_total: float = 0.0  # COUNTER
    kv_evictable_tokens: float = 0.0
    mamba_evictable_tokens: float = 0.0
    hicache_host_used_tokens: float = 0.0
    hicache_host_total_tokens: float = 0.0
    # vLLM prefix-cache counters (cumulative); the client-side
    # usage.prompt_tokens_details.cached_tokens path is null on current
    # vLLM builds even when caching is active, so the ratio is derived
    # from these counter deltas instead.
    prefix_cache_queries_total: float = 0.0
    prefix_cache_hits_total: float = 0.0
