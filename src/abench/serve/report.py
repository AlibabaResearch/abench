# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Human-readable and JSON reporting for agentic replay metrics."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional

from abench.core.metrics import AgenticMetrics, BucketStats, SessionRecord


def _line(width: int = 78) -> str:
    return "=" * width


def _section(title: str) -> None:
    print(f"------{title}------")


def _kv(label: str, value: Any) -> None:
    print(f"{label:<40}{value}")


def _fmt_bucket(b: BucketStats) -> str:
    return (
        f"  {b.label:<10} n={b.count:<6d} "
        f"mean={b.mean_ms:8.1f}ms  p50={b.p50_ms:8.1f}ms  "
        f"p90={b.p90_ms:8.1f}ms  p95={b.p95_ms:8.1f}ms  p99={b.p99_ms:8.1f}ms"
    )


def print_report(m: AgenticMetrics, config: Dict[str, Any]) -> None:
    print()
    print(_line())
    print("ABench — closed-loop agentic inference benchmark")
    print(_line())

    # Header summary, bench_serving style. Whole-run roll-up: every
    # successful turn of the run, wall clock first dispatch -> last
    # completion. The throughput rows further down are computed over the
    # steady window instead.
    _kv("Backend", config.get("backend", ""))
    _kv("dataset_name", config.get("dataset_name", ""))
    if config.get("dataset_path"):
        _kv("dataset_path", config["dataset_path"])
    if config.get("tag"):
        _kv("tag", config["tag"])
    _kv("target concurrency", m.target_concurrency)
    _kv("successful turns", m.successful_turns)
    _kv("benchmark duration (s)", f"{m.wall_duration_s:.2f}")
    _kv("total input tokens", f"{m.total_input_tokens:,}")
    _kv("total output tokens", f"{m.total_output_tokens:,}")
    _kv("turns throughput (turns/s)", f"{m.turns_throughput:.2f}")
    _kv("input token throughput (tok/s)", f"{m.input_throughput:.1f}")
    _kv("output token throughput (tok/s)", f"{m.output_throughput:.1f}")
    _kv("peak output throughput (1s, tok/s)", f"{m.peak_output_throughput:.1f}")
    _kv("concurrency mean", f"{m.concurrency_mean:.2f}")
    _kv("concurrency peak", m.concurrency_peak)
    print()

    print("Configuration")
    for k in (
        "model",
        "backend",
        "dataset_name",
        "dataset_path",
        "tag",
        "system_prompt_len",
        "max_concurrency",
        "duration_s",
    ):
        if k in config and config[k] is not None:
            print(f"  {k:<20} {config[k]}")
    print()

    print("Load / steady-state window")
    print(f"  target concurrency   {m.target_concurrency}")
    print(f"  steady threshold     >= {m.steady_frac:.2f} x C")
    print(f"  turns total          {m.total_turns}")
    print(f"  turns in window      {m.steady_turns}")
    print(f"  steady window        {m.steady_window_s:.2f} s")
    if m.concurrency_hist:
        hist = ", ".join(
            f"{c}:{n}" for c, n in sorted(m.concurrency_hist.items())
        )
        print(f"  achieved concurrency {{{hist}}}")
    print()

    _section("End-to-End Latency")
    if m.e2e_overall:
        print("  (whole run)")
        print(_fmt_bucket(m.e2e_overall))
    if m.e2e_by_turn:
        print("  by turn bucket (steady window)")
        for b in m.e2e_by_turn:
            print(_fmt_bucket(b))
    if m.e2e_by_inputlen:
        print("  by input-length bucket (steady window)")
        for b in m.e2e_by_inputlen:
            print(_fmt_bucket(b))
    print()

    _section("TTFT")
    if m.t1_ttft:
        print(_fmt_bucket(m.t1_ttft))
    if m.t2plus_ttft:
        print(_fmt_bucket(m.t2plus_ttft))
    if m.t1_ttft and m.t1_ttft.count > 0:
        print(f"  TTFT degradation (T2+/T1): {m.ttft_degradation:.3f}x")
    else:
        print("  TTFT degradation (T2+/T1): n/a (no T1 turns in window)")
    print()

    _section("TPOT")
    print(f"  TPOT (mean ITL):           {m.tpot_ms:.2f} ms")
    print()

    _section("ITL")
    if m.itl_max_ms > 0:
        print(
            f"  ITL p50/p90/p95/p99/max:   {m.itl_p50_ms:.1f} / "
            f"{m.itl_p90_ms:.1f} / {m.itl_p95_ms:.1f} / {m.itl_p99_ms:.1f} / "
            f"{m.itl_max_ms:.1f} ms"
        )
    else:
        print("  ITL: n/a (no inter-token samples)")
    print()

    _section("Cache / Throughput (steady window)")
    print(f"  prefix cache ratio    {m.prefix_cache_ratio * 100:.1f} %")
    if m.host_cache_ratio:
        print(
            f"    of which host (L2)  {m.host_cache_ratio * 100:.1f} % of input"
            f"  ({m.host_share_of_cache * 100:.1f} % of all hits)"
        )
    print(f"  output throughput     {m.output_throughput:.1f} tok/s")
    print(f"  total throughput      {m.total_throughput:.1f} tok/s")
    print()

    print("Pareto point (plot y against x across concurrency levels)")
    print(f"  y system output       {m.output_throughput:.1f} tok/s")
    print(
        f"  x per-user output     {m.per_user_output_tps:.2f} tok/s"
        "   (tool waits excluded)"
    )
    print()

    print("Output length fidelity")
    if m.osl_mean_expected > 0:
        lo = m.osl_mean_expected * (1.0 - m.osl_tolerance)
        hi = m.osl_mean_expected * (1.0 + m.osl_tolerance)
        print(
            f"  mean OSL              {m.osl_mean_achieved:.1f} "
            f"(recorded {m.osl_mean_expected:.1f}, "
            f"band [{lo:.0f},{hi:.0f}] at +/-{m.osl_tolerance * 100:.0f}%)"
        )
        print(f"  within band           {'YES' if m.osl_within_band else 'NO'}")
        if not m.osl_within_band and m.osl_mean_achieved > 4 * m.osl_mean_expected:
            print(
                "  note                  achieved mean is near the generation "
                "cap: the model never chose to stop, so this workload is far "
                "heavier than the trace."
            )
    else:
        print("  mean OSL              n/a (no recorded lengths in window)")
    print()

    _section("Session")
    print(f"  sessions              {m.num_sessions}")
    print(f"  completed sessions    {m.completed_sessions}")
    if m.truncated_sessions:
        print(
            f"  truncated sessions    {m.truncated_sessions} "
            f"(stopped by pool exhaustion; excluded from stats)"
        )
    if m.ctx_guard_sessions:
        print(
            f"  ctx-guard stops       {m.ctx_guard_sessions} "
            f"(estimated prompt + pinned completion hit max_model_len)"
        )
    print(f"  turn completion rate  {m.completion_rate * 100:.1f} %")
    print(f"  mean trajectory time  {m.mean_trajectory_time_s:.2f} s")
    print()
    if m.routing_stickiness is not None:
        _section("Routing")
        print(f"  routing stickiness    {m.routing_stickiness * 100:.1f} %")
        print()
    if m.kv_snapshots_count > 0:
        _section("KV Pool (server-side Prometheus)")
        print(f"  snapshots             {m.kv_snapshots_count}")
        if m.kv_pool_capacity_tokens is not None:
            print(f"  pool capacity         {m.kv_pool_capacity_tokens:,.0f} tokens")
        if m.kv_pool_utilization_mean is not None:
            print(
                f"  utilization           mean={m.kv_pool_utilization_mean * 100:.1f}%  "
                f"p99={(m.kv_pool_utilization_p99 or 0.0) * 100:.1f}%"
            )
        if m.kv_eviction_delta is not None:
            print(f"  evictions (run)       {m.kv_eviction_delta:,.0f}")
        if m.kv_queue_depth_p99 is not None:
            print(
                f"  queue depth           p99={m.kv_queue_depth_p99:.0f}  "
                f"running mean={m.kv_running_mean if m.kv_running_mean is not None else 0.0:.1f}"
            )
        print(_line())


def _bucket_to_dict(b: BucketStats) -> Dict[str, Any]:
    return {
        "label": b.label,
        "count": b.count,
        "mean_ms": b.mean_ms,
        "p50_ms": b.p50_ms,
        "p90_ms": b.p90_ms,
        "p95_ms": b.p95_ms,
        "p99_ms": b.p99_ms,
    }


def metrics_to_dict(m: AgenticMetrics) -> Dict[str, Any]:
    return {
        "target_concurrency": m.target_concurrency,
        "steady_frac": m.steady_frac,
        "total_turns": m.total_turns,
        "steady_turns": m.steady_turns,
        "steady_window_s": m.steady_window_s,
        "concurrency_hist": m.concurrency_hist,
        "t1_ttft": _bucket_to_dict(m.t1_ttft) if m.t1_ttft else None,
        "t2plus_ttft": _bucket_to_dict(m.t2plus_ttft) if m.t2plus_ttft else None,
        "tpot_ms": m.tpot_ms,
        "itl_p50_ms": m.itl_p50_ms,
        "itl_p90_ms": m.itl_p90_ms,
        "itl_p95_ms": m.itl_p95_ms,
        "itl_p99_ms": m.itl_p99_ms,
        "itl_max_ms": m.itl_max_ms,
        "ttft_degradation": m.ttft_degradation,
        "e2e_by_turn": [_bucket_to_dict(b) for b in m.e2e_by_turn],
        "e2e_by_inputlen": [_bucket_to_dict(b) for b in m.e2e_by_inputlen],
        "prefix_cache_ratio": m.prefix_cache_ratio,
        "host_cache_ratio": m.host_cache_ratio,
        "host_share_of_cache": m.host_share_of_cache,
        "output_throughput": m.output_throughput,
        "total_throughput": m.total_throughput,
        "per_user_output_tps": m.per_user_output_tps,
        "osl_mean_achieved": m.osl_mean_achieved,
        "osl_mean_expected": m.osl_mean_expected,
        "osl_tolerance": m.osl_tolerance,
        "osl_within_band": m.osl_within_band,
        "num_sessions": m.num_sessions,
        "completed_sessions": m.completed_sessions,
        "truncated_sessions": m.truncated_sessions,
        "ctx_guard_sessions": m.ctx_guard_sessions,
        "completion_rate": m.completion_rate,
        "mean_trajectory_time_s": m.mean_trajectory_time_s,
        "routing_stickiness": m.routing_stickiness,
        "kv_pool_capacity_tokens": m.kv_pool_capacity_tokens,
        "kv_pool_utilization_mean": m.kv_pool_utilization_mean,
        "kv_pool_utilization_p99": m.kv_pool_utilization_p99,
        "kv_eviction_delta": m.kv_eviction_delta,
        "kv_queue_depth_p99": m.kv_queue_depth_p99,
        "kv_running_mean": m.kv_running_mean,
        "kv_snapshots_count": m.kv_snapshots_count,
        # whole-run roll-up (bench_serving-style header); JSONL-compatible
        # append — older rows simply lack these keys.
        "successful_turns": m.successful_turns,
        "wall_duration_s": m.wall_duration_s,
        "total_input_tokens": m.total_input_tokens,
        "total_output_tokens": m.total_output_tokens,
        "turns_throughput": m.turns_throughput,
        "input_throughput": m.input_throughput,
        "peak_output_throughput": m.peak_output_throughput,
        "e2e_overall": _bucket_to_dict(m.e2e_overall) if m.e2e_overall else None,
        "concurrency_mean": m.concurrency_mean,
        "concurrency_peak": m.concurrency_peak,
    }


def save_results(
    sessions: List[SessionRecord],
    metrics: AgenticMetrics,
    path: str,
    config: Dict[str, Any],
    server_snapshots: Optional[List[Any]] = None,
    server_info: Optional[dict] = None,
    include_details: bool = True,
) -> None:
    """Append one JSON object per line (JSONL) to ``path``.

    ``server_info`` is the SGLang ``/server_info`` snapshot (None when the
    backend does not expose it). ``include_details`` adds the per-turn
    records; without it the line carries only config + metrics.
    """
    payload = {
        "config": config,
        "server_info": server_info,
        "metrics": metrics_to_dict(metrics),
        "server_metrics": [
            {
                "t_offset_s": round(s.timestamp - server_snapshots[0].timestamp, 3)
                if server_snapshots
                else 0,
                **{
                    f.name: getattr(s, f.name)
                    for f in dataclasses.fields(type(s))
                    if f.name != "timestamp"
                },
            }
            for s in (server_snapshots or [])
        ],
    }
    if include_details:
        payload["sessions"] = [
            {
                "session_id": s.session_id,
                "source_model": s.source_model,
                "assigned_workers": s.assigned_workers,
                "truncated": s.truncated,
                "trajectory_inference_time": s.trajectory_inference_time,
                "turns": [
                    {
                        "turn_index": t.turn_index,
                        "input_len": t.input_len,
                        "output_len": t.output_len,
                        "recorded_output_len": t.recorded_output_len,
                        "ttft": t.ttft,
                        "latency": t.latency,
                        "server_cached_tokens": t.server_cached_tokens,
                        "cached_device_tokens": t.cached_device_tokens,
                        "cached_host_tokens": t.cached_host_tokens,
                        "active_at_dispatch": t.active_at_dispatch,
                        "success": t.success,
                        "error": t.error,
                        # wall-clock pair, to align a turn with server-side
                        # traces and monitoring windows
                        "start_ts": t.dispatch_epoch,   # epoch seconds
                        "end_ts": t.complete_epoch,     # epoch seconds
                        "itl_ms": [round(x * 1000.0, 3) for x in t.itl],
                        "tpot_ms": (
                            round((t.latency - t.ttft) / (t.output_len - 1) * 1000.0, 3)
                            if t.output_len > 1 and t.success
                            else None
                        ),
                        "status": (
                            "ok"
                            if t.success
                            else (
                                "timeout"
                                if any(
                                    k in (t.error or "")
                                    for k in ("Timeout", "timeout", "TimeoutError")
                                )
                                else "err"
                            )
                        ),
                    }
                    for t in s.turns
                ],
            }
            for s in sessions
        ]
    with open(path, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
