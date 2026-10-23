# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Closed-loop agentic replay engine.

Implements the concurrency model from the benchmark design:

* A fixed number of **active sessions** ``C`` is maintained. Each of the
  ``C`` async workers holds exactly one session at a time and drives it
  turn-by-turn (a session is strictly sequential: turn N+1 waits for
  turn N to complete). When a session finishes, the worker pulls the
  next one from the pool — so a short trajectory is simply *recycled
  faster*, keeping the active count pinned at ``C`` without any turn
  truncation.

* ``--duration`` (optional) turns the pool into a looping source and
  stops dispatching new sessions once the deadline passes; in-flight
  sessions are allowed to finish. This removes the drain phase from the
  measurement window.

* **Tool waits** (``pre_gap_ms``) are honored *between* turns but are not
  counted in per-turn latency — the model is idle, not the client.

* Every turn's decode length is pinned by the output policy — the
  recorded ``completion_tokens`` when replaying a trace, a constant when
  the workload is synthetic — and sent as ``max_completion_tokens`` with
  ``ignore_eos``. The model's own EOS never decides how much work a turn
  does, which is what makes two engines comparable on the same workload.

HTTP plumbing lives in ``abench.replay.transport``, server discovery and
telemetry in ``abench.serve``.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from abench.core.metrics import ServerMetricsSnapshot, SessionRecord, TurnRecord
from abench.core.workload import PlannedSession
from abench.replay.transport import (
    _count_tokens_async,
    _process_non_stream,
    _process_stream,
)
from abench.serve.endpoints import get_auth_headers, server_root
from abench.serve.telemetry import _scrape_server_metrics

# ----------------------------------------------------------------------
# Shared run state (concurrency accounting + pool)
# ----------------------------------------------------------------------
class _RunState:
    """Thread/async-safe dispatch state shared by all workers."""

    def __init__(
        self,
        pool: List[PlannedSession],
        loop_pool: bool,
        deadline: Optional[float],
        max_duration_s: Optional[float] = None,
    ) -> None:
        self._pool = pool
        self._loop = loop_pool
        self._deadline = deadline
        # Runaway fuse: hard cap on total run wall clock, independent of
        # any window. Guards against a wedged server or endless retry
        # dragging a run on forever; a normal run should never reach it.
        self._max_deadline = (
            time.perf_counter() + max_duration_s if max_duration_s else None
        )
        self._idx = 0
        self._lock = threading.Lock()
        self.active = 0
        self.exhausted = False
        self.planned_turns = sum(len(s.turns) for s in pool)
        self.turns_done = 0
        self.output_tokens = 0
        # Optional tqdm handle; set by run_closed_loop when a progress bar
        # is active. Workers update it per completed turn.
        self.pbar = None
        self.sessions_done = 0
        self.sessions_truncated = 0

    def effective_deadline(self) -> Optional[float]:
        """Earliest of the --duration window and the --max-duration cap."""
        if self._deadline is not None and self._max_deadline is not None:
            return min(self._deadline, self._max_deadline)
        return self._deadline if self._deadline is not None else self._max_deadline

    def next_session(self) -> Optional[PlannedSession]:
        d = self.effective_deadline()
        if d is not None and time.perf_counter() >= d:
            return None
        with self._lock:
            if self._idx >= len(self._pool):
                if not self._loop:
                    self.exhausted = True
                    return None
                self._idx = 0
            sess = self._pool[self._idx]
            self._idx += 1
            return sess


# ----------------------------------------------------------------------
# Single-session driver
# ----------------------------------------------------------------------
def _turn_request_timeout(
    turn_timeout_s: float, deadline: Optional[float], now: float
) -> Optional[aiohttp.ClientTimeout]:
    """Per-request cap: the tighter of the turn timeout and the remaining
    ``--max-duration`` budget. The fuse's between-turn check cannot stop a
    hung in-flight request; without this, one wedged turn holds the run
    hostage far past the deadline. ``None``
    means "no deadline": fall back to the session's default timeout."""
    if deadline is None:
        return None
    remaining = deadline - now
    if remaining <= 0:
        return None
    return aiohttp.ClientTimeout(total=min(turn_timeout_s, remaining))


async def _run_session(
    sess: PlannedSession,
    state: _RunState,
    api_url: str,
    model: str,
    session: aiohttp.ClientSession,
    tokenizer: Optional[Any],
    content_mode: str,
    disable_stream: bool,
    extra_body: Dict[str, Any],
    respect_gaps: bool,
    gap_scale: float = 1.0,
    ignore_eos: bool = True,
    worker_header: Optional[str] = None,
    session_timeout_s: float = 0.0,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_request_body: Optional[Dict[str, Any]] = None,
    backend: str = "sglang",
    max_model_len: Optional[int] = None,
    turn_timeout_s: float = 3600.0,
) -> SessionRecord:
    rec = SessionRecord(
        session_id=sess.session_id,
        source_model=str(sess.metadata.get("model", "")),
    )

    session_start = time.perf_counter()

    messages: List[Dict[str, str]] = []
    # Running client-side estimate of the prompt length. Each piece of
    # text is encoded exactly once (incremental), and the server-reported
    # ``usage.prompt_tokens`` overrides the estimate whenever available.
    input_len_est = 0
    if sess.system_prompt:
        messages.append({"role": "system", "content": sess.system_prompt})
        input_len_est += await _count_tokens_async(sess.system_prompt, tokenizer)

    for turn in sess.turns:
        # Timed runs: truncate in-flight sessions once the window closes
        # instead of draining their whole trajectory. Drain turns are
        # excluded from steady-window statistics anyway, so keeping them
        # only stretches the wall clock (a timed run can last 5-10x the
        # window on long trajectories with tool gaps).
        d = state.effective_deadline()
        if d is not None and time.perf_counter() > d:
            rec.truncated = True
            break

        # Per-session timeout: stop dispatching new turns once the budget
        # is spent. Already-completed turns stay; the session is marked
        # truncated so it is excluded from trajectory stats.
        if session_timeout_s > 0 and (time.perf_counter() - session_start) >= session_timeout_s:
            rec.truncated = True
            break

        # Client-side context guard: estimated prompt plus the pinned
        # completion must fit the model's context window, else the server
        # rejects the turn with an HTTP 400. Later turns only grow, so
        # stop the session instead of burning guaranteed failures.
        if (
            max_model_len is not None
            and input_len_est + turn.max_new_tokens > max_model_len
        ):
            rec.truncated = True
            rec.ctx_guard_stop = True
            break

        if respect_gaps and turn.pre_gap_ms > 0:
            await asyncio.sleep(turn.pre_gap_ms * gap_scale / 1000.0)

        for m in turn.input_delta:
            # Tool observations are recorded with role="tool" but carry no
            # tool_call_id, which strict OpenAI-schema validators (routers)
            # reject. Role does not affect prefill/decode cost, so coerce any
            # non-standard role to "user" for schema-safe replay.
            role = m["role"] if m["role"] in ("system", "user", "assistant") else "user"
            messages.append({"role": role, "content": m["content"]})
            input_len_est += await _count_tokens_async(m["content"], tokenizer)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": turn.max_new_tokens,
            "stream": not disable_stream,
        }
        if ignore_eos:
            payload["ignore_eos"] = True
        if not disable_stream:
            payload["stream_options"] = {"include_usage": True}
        if backend == "sglang":
            # Off by default server-side; without it the device/host split
            # of prefix-cache hits is not reported at all. SGLang-only:
            # other backends may reject unknown request fields outright.
            payload["return_cached_tokens_details"] = True
        payload.update(extra_body or {})
        # --extra-request-body goes last so it can override anything above.
        payload.update(extra_request_body or {})

        headers = {"Content-Type": "application/json"}
        headers.update(get_auth_headers())
        headers.update(extra_headers or {})
        if sess.routing_key:
            headers["X-SMG-Routing-Key"] = sess.routing_key

        tr = TurnRecord(
            session_id=sess.session_id,
            turn_index=turn.index,
            input_len=input_len_est,
            active_at_dispatch=state.active,
            recorded_output_len=turn.recorded_output_tokens,
        )
        start = time.perf_counter()
        tr.dispatch_time = start
        tr.dispatch_epoch = time.time()  # exported as start_ts (epoch s)
        gen_text = ""
        try:
            req_timeout = _turn_request_timeout(
                turn_timeout_s, state.effective_deadline(), time.perf_counter()
            )
            async with session.post(
                api_url, json=payload, headers=headers, timeout=req_timeout
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    tr.success = False
                    tr.error = f"HTTP {resp.status}: {body[:200]}"
                else:
                    if disable_stream:
                        gen_text, ttft, itl, usage = await _process_non_stream(
                            resp, start
                        )
                    else:
                        gen_text, ttft, itl, usage = await _process_stream(
                            resp, start
                        )
                    tr.ttft = ttft
                    tr.itl = itl
                    tr.server_cached_tokens = usage["cached_tokens"]
                    tr.cached_device_tokens = usage["cached_device_tokens"]
                    tr.cached_host_tokens = usage["cached_host_tokens"]
                    # Server-reported token counts are authoritative (they
                    # include chat-template overhead the client can't see).
                    if usage["prompt_tokens"] > 0:
                        tr.input_len = usage["prompt_tokens"]
                    if usage["completion_tokens"] > 0:
                        tr.output_len = usage["completion_tokens"]
                    else:
                        tr.output_len = await _count_tokens_async(
                            gen_text, tokenizer
                        )
                    tr.success = bool(gen_text) or tr.output_len > 0
                    if worker_header:
                        worker = resp.headers.get(worker_header)
                        if worker:
                            rec.assigned_workers.append(worker)
        except Exception as exc:  # noqa: BLE001
            tr.success = False
            tr.error = f"{type(exc).__name__}: {exc} | " + traceback.format_exc(
                limit=1
            )
        tr.complete_time = time.perf_counter()
        tr.complete_epoch = time.time()  # exported as end_ts (epoch s)
        tr.latency = tr.complete_time - start
        rec.turns.append(tr)
        state.turns_done += 1
        state.output_tokens += tr.output_len
        if state.pbar is not None:
            state.pbar.update(1)

        # Advance the running context with the assistant slot.
        if content_mode == "model" and tr.success:
            messages.append({"role": "assistant", "content": gen_text})
            input_len_est += tr.output_len
        else:
            messages.append({"role": "assistant", "content": turn.assistant_text})
            input_len_est += await _count_tokens_async(
                turn.assistant_text, tokenizer
            )

    return rec


# ----------------------------------------------------------------------
# Closed-loop orchestration
# ----------------------------------------------------------------------
async def run_closed_loop(
    pool: List[PlannedSession],
    api_url: str,
    model: str,
    max_concurrency: int,
    tokenizer: Optional[Any],
    content_mode: str = "trace",
    disable_stream: bool = False,
    duration_s: Optional[float] = None,
    respect_gaps: bool = True,
    gap_scale: float = 1.0,
    extra_body: Optional[Dict[str, Any]] = None,
    worker_header: Optional[str] = None,
    verbose: bool = False,
    progress_interval: float = 30.0,
    ignore_eos: bool = True,
    turn_timeout_s: float = 3600.0,
    metrics_interval: float = 5.0,
    session_timeout_s: float = 0.0,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_request_body: Optional[Dict[str, Any]] = None,
    backend: str = "sglang",
    max_duration_s: Optional[float] = None,
    disable_tqdm: bool = False,
    max_model_len: Optional[int] = None,
) -> Tuple[List[SessionRecord], List[ServerMetricsSnapshot]]:
    """Drive ``pool`` with ``concurrency`` active sessions.

    When ``duration_s`` is set the pool loops and dispatch stops at the
    deadline; otherwise every session is played exactly once.
    ``max_duration_s`` is a hard cap on top of either mode (runaway fuse).

    ``backend`` gates the observation layer: only ``sglang`` sends
    ``return_cached_tokens_details`` and scrapes ``/metrics`` with the
    ``sglang:`` gauge names; ``vllm`` scrapes with ``vllm:`` names;
    ``openai`` skips scraping entirely.

    Returns ``(session_records, server_metrics_snapshots)``. The snapshots
    list is empty when the server does not expose ``/metrics``.
    """
    deadline = (time.perf_counter() + duration_s) if duration_s else None
    state = _RunState(
        pool,
        loop_pool=duration_s is not None,
        deadline=deadline,
        max_duration_s=max_duration_s,
    )
    results: List[SessionRecord] = []
    results_lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=max(max_concurrency * 2, 32), ssl=False)
    # Applies per request: a hung turn frees its slot after this long instead
    # of holding it for the rest of the run.
    timeout = aiohttp.ClientTimeout(total=turn_timeout_s)

    async def progress() -> None:
        """Heartbeat so a multi-hour run is distinguishable from a stall."""
        t0 = time.perf_counter()
        while True:
            await asyncio.sleep(heartbeat_interval)
            el = time.perf_counter() - t0
            rate = state.output_tokens / el if el > 0 else 0.0
            line = (
                f"[progress] {el / 60:6.1f}min  "
                f"turns {state.turns_done}/{state.planned_turns} "
                f"({100.0 * state.turns_done / max(1, state.planned_turns):4.1f}%)  "
                f"sessions done={state.sessions_done} trunc={state.sessions_truncated}  "
                f"active={state.active}/{max_concurrency}  "
                f"out={rate:7.1f} tok/s"
            )
            if state.planned_turns and state.turns_done:
                remaining = state.planned_turns - state.turns_done
                eta_min = remaining / (state.turns_done / el) / 60.0
                line += f"  eta~{eta_min:5.1f}min"
            print(line, flush=True)

    server_snapshots: List[ServerMetricsSnapshot] = []
    metrics_url = server_root(api_url) + "/metrics"

    # Optional turn-level progress bar. tqdm is not a hard dependency: with
    # it absent (or disable_tqdm set) the periodic heartbeat above remains
    # the only progress signal, exactly as before.
    pbar = None
    if not disable_tqdm:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(
                total=state.planned_turns,
                unit="turn",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
        except ImportError:
            pbar = None
    state.pbar = pbar
    # With the bar live the heartbeat is redundant visual noise unless the
    # interval is explicitly raised; keep it but at a coarser cadence.
    heartbeat_interval = progress_interval if pbar is None else max(
        progress_interval, 300.0
    )

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http:

        async def worker(wid: int) -> None:
            while True:
                sess = state.next_session()
                if sess is None:
                    return
                state.active += 1
                try:
                    rec = await _run_session(
                        sess=sess,
                        state=state,
                        api_url=api_url,
                        model=model,
                        session=http,
                        tokenizer=tokenizer,
                        content_mode=content_mode,
                        disable_stream=disable_stream,
                        extra_body=extra_body or {},
                        respect_gaps=respect_gaps,
                        gap_scale=gap_scale,
                        ignore_eos=ignore_eos,
                        worker_header=worker_header,
                        session_timeout_s=session_timeout_s,
                        extra_headers=extra_headers,
                        extra_request_body=extra_request_body,
                        backend=backend,
                        max_model_len=max_model_len,
                        turn_timeout_s=turn_timeout_s,
                    )
                finally:
                    state.active -= 1
                if rec.truncated:
                    state.sessions_truncated += 1
                else:
                    state.sessions_done += 1
                async with results_lock:
                    results.append(rec)
                if verbose:
                    print(
                        f"[w{wid}] {sess.session_id[:32]} "
                        f"turns={len(rec.turns)} active={state.active}"
                    )

        workers = [asyncio.create_task(worker(i)) for i in range(max_concurrency)]
        reporter = (
            asyncio.create_task(progress()) if progress_interval > 0 else None
        )
        # The generic openai backend has no known /metrics gauges to read.
        scraper = (
            asyncio.create_task(
                _scrape_server_metrics(
                    http,
                    metrics_url,
                    metrics_interval,
                    server_snapshots,
                    verbose,
                    backend=backend,
                    headers=get_auth_headers(),
                )
            )
            if metrics_interval > 0 and backend in ("sglang", "vllm")
            else None
        )
        try:
            await asyncio.gather(*workers)
        finally:
            if reporter is not None:
                reporter.cancel()
            if scraper is not None:
                scraper.cancel()
            if pbar is not None:
                pbar.close()

    if max_model_len is not None:
        stopped = sum(1 for r in results if r.ctx_guard_stop)
        if stopped:
            print(
                f"[ctx-guard] stopped {stopped} session(s) early: estimated "
                f"prompt + pinned completion exceeded the model context "
                f"({max_model_len} tokens)",
                flush=True,
            )
    return results, server_snapshots


# Warmup exists to prime the shared prefix and surface config errors, not
# to pre-run whole trajectories: the prefix-cache benefit concentrates in
# the first blocks, and a full long session replayed at concurrency 1 can
# outlast the measured run and hand that session a fully warm cache into
# the measurement (found in large-scale real-server testing).
WARMUP_MAX_TURNS = 3


async def warmup(
    pool: List[PlannedSession],
    api_url: str,
    model: str,
    tokenizer: Optional[Any],
    n: int,
    disable_stream: bool = False,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_request_body: Optional[Dict[str, Any]] = None,
    backend: str = "sglang",
    max_duration_s: Optional[float] = None,
) -> None:
    """Play the first ``n`` sessions once to populate the prefix cache.

    Each session is truncated to its first ``WARMUP_MAX_TURNS`` turns and
    the whole phase is bounded by ``max_duration_s`` (the same runaway fuse
    the measured run gets), so warmup can never dominate wall clock.
    """
    if n <= 0:
        return
    trimmed: List[PlannedSession] = []
    for sess in pool[:n]:
        if len(sess.turns) > WARMUP_MAX_TURNS:
            sess = replace(sess, turns=sess.turns[:WARMUP_MAX_TURNS])
        trimmed.append(sess)
    await run_closed_loop(
        pool=trimmed,
        api_url=api_url,
        model=model,
        max_concurrency=min(n, len(pool)) or 1,
        tokenizer=tokenizer,
        content_mode="trace",
        disable_stream=disable_stream,
        respect_gaps=False,
        # Warmup runs at its own tiny concurrency over a handful of
        # sessions; reporting its progress would look like the real run's.
        progress_interval=0,
        metrics_interval=0,  # no scraping during warmup
        extra_headers=extra_headers,
        extra_request_body=extra_request_body,
        backend=backend,
        max_duration_s=max_duration_s,
    )
