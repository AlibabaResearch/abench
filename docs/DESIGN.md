# Design — ABench: Closed-loop Agentic Inference Benchmark

This benchmark measures **how fast a model service executes a fixed
amount of agent-inference work**. It replays real
[DeepSWE v1.1](https://deepswe.datacurve.ai/data/v1.1/trials) agent
trajectories in a closed loop. It is an *inference performance*
benchmark, not an agent-capability leaderboard: tool correctness and
task success are irrelevant; only serving speed and cache behavior are.

It deliberately mirrors the CLI ergonomics of SGLang's
`bench_serving.py` — sampling defaults, environment-based auth, JSONL
output — while replacing the open-loop request stream with closed-loop
replay of whole agent sessions.

## Pipeline

```
ModelScope mirror ──download──▶ sessions (.jsonl.zst) ──select+plan──▶ closed-loop replay ──▶ metrics
 (Apsara-Stack/ABench)           (SHA-256 verified,       (knobs applied)  (C active sessions)  (steady window)
                                  one line = one session)
```

- **`core/dataset.py`** holds the tier catalog (`TIERS`, hardcoded to the
  published mirror — `deepswe/mixed`, `deepswe/full`, `lmcache/full`)
  and downloads each tier's `*.jsonl.zst` parts from the ModelScope
  resolve endpoint: streaming, resumable via HTTP `Range` on a `.part`
  file, and SHA-256 verified against the catalog before the atomic
  rename. Loading is streaming too (`zstandard` package if installed,
  else the `zstd` CLI), so the 22K-session `full` tier never needs to be
  fully decompressed on disk. The JSONL schema (`abench-session-jsonl-v1`)
  carries the *authoritative* token counts the source runs reported
  (`prompt_tokens`, `completion_tokens`, `cached_tokens`), tool
  observations as the next turn's input delta, and `pre_gap_ms`.

  The normalization that produced the mirror happened once at packaging
  time (DeepSWE trajectories: system prompt stripped into its own field;
  LMCache Otel traces: message-diff per turn) and is not part of the
  client: users only ever download the finished, checksummed artifacts.

  `pre_gap_ms` in the packaged data is the interval between consecutive
  *agent step* timestamps, which is a full cycle: tool execution plus one
  turn of the recording model's generation. It is the generation of the
  turn the interval leads *into* (interval correlates with the next
  turn's output length at r=0.68, against r=0.16 for the previous turn's).
  Sleeping it in full would charge the replayed server for inference
  twice, so `workload.tool_delays_ms` subtracts
  `output_tokens / rate`, where `rate` comes from
  `data/recording_rates.json` — the 5th percentile of
  `interval / output_tokens` fitted per source model over the whole
  corpus (18 models, 1,549,994 intervals; 59-293 tok/s, pooled 140).
  On the small band this turns a 5.7 s median interval into a 3.1 s
  median tool delay, removing 38% of total idle and zeroing 4% of turns.

  The split is a modelling assumption, not a measurement: the dataset has
  no per-tool duration field, and the one independent probe — turn 0,
  which has no preceding tool call — is unusable because non-agent steps
  carry whole-second timestamps only. Three properties make it
  defensible. It is *bounded*: every fitted rate exceeds
  `n_output_tokens / agent_duration_seconds` from the trial index by
  1.6-3.6x, as it must, since that ratio charges tool time to generation.
  It is *conservative*: the 5th percentile assumes the fastest turns
  carry no tool work, so the rate is an upper bound and the removed
  generation time a lower bound — the split never eats into real tool
  idle. And it is *server-independent*, so the same trajectory offers
  identical idle time to every server under test, which is what makes
  cross-model runs comparable and is the property MLPerf gets for free
  from its dataset-side `delay_seconds` field. Sensitivity to the
  percentile is real and worth quoting with results: refitting the table
  at the 2nd percentile leaves a 4.0 s median delay (26% of idle removed)
  and at the 10th a 2.3 s median (47% removed). Fitting per
  session instead of per model is *not* viable — within one session tool
  time correlates with output length, so the regression is not
  identified; a Theil-Sen variant returned 32 tok/s at the 10th
  percentile of small-band sessions and zeroed 26% of turns.

  The rates table was fitted at packaging time; it ships with the
  package (`data/recording_rates.json`). `assistant_text` folds in
  `reasoning_content` as well as `tool_calls`, because the authoritative
  `completion_tokens` counts reasoning (up to 57% of a turn's output for
  some source models) and dropping it would replay a shorter context than
  the run being reproduced.

  Two external references bound the estimate. MLPerf does not estimate at
  all: its `delay_seconds` is a measured per-tool duration, visible in the
  public edge dataset where it appears only on `tool` rows with a median
  of 0.06 s, a mean of 0.64 s and a maximum of 120 s. The DeepSWE
  artifacts expose no equivalent field — one timestamp per agent step, no
  timestamp on the observation, and `agent_log` is 403 upstream — and
  the prepared `agentic_combined.jsonl` is not published, so the estimate
  cannot simply be replaced by their value. Independently, NVIDIA's
  synthetic agentic-code generator models tool-call follow-up delay as
  lognormal with a median of 1.8 s and a mean of 2.5 s; that is the order
  of the 3.1 s median extracted here, and not the order of the 5.7 s raw
  interval.
- **`core/workload.py`** applies the four knobs and freezes the mixed
  profile.
- **`replay/`** runs the closed loop: `engine.py` schedules sessions
  and owns warmup and the run fuse; `transport.py` parses each turn's
  streaming / non-streaming response. Trace replay is the only content
  policy.
- **`core/metrics.py`** does steady-state windowing + per-bucket
  aggregation.
- **`serve/`** isolates everything that talks to the server under
  test: endpoint/backend discovery, auth, readiness and cache flush
  (`endpoints.py`), `/metrics` scraping and Prometheus parsing
  (`telemetry.py`), terminal and JSONL reporting (`report.py`).
- **`cli.py`** parses arguments; **`runner.py`** orchestrates a `data`
  or `run` command end to end.

## The four design decisions

### 1. System-prompt length is a parameter

The recorded system prompt is stripped at normalization time. At replay,
`build_system_prompt(target_tokens)` tiles a fixed canonical block to an
**exact token length** (via the tokenizer). Rationale: for performance,
only the token count of the shared static prefix matters, not its
content — and the model output is discarded anyway. The synthesized
prompt is **byte-identical across all sessions**, so its KV is fully
cacheable, matching production where the system prompt is shared.
`--system-prompt-len 0` disables it; compare runs with e.g. `1024,4096,16384`.

### 2. Turns / context / output are sampling knobs

- `--min-turns` / `--max-turns`: a **filter + cap**, never padding.
  Short sessions are dropped below `min_turns`; long sessions are
  truncated only when `max_turns` is set. We never invent turns.
- `--min-context` / `--max-context`: an **explicit token band** on a
  session's replayed peak context — the recorded peak plus the measured
  length of the injected shared system prompt — (inclusive, tested after any
  `max_turns` cap). Both edges are stated rather than derived from a
  center and a tolerance, so it is obvious which sessions a band admits.
  `--max-context` is user-facing in **model-window** terms: the value is
  the model's context window, and the tool derives the recorded-peak cap
  as `max_context - system prompt` (printed on the `[plan]` line), so
  users never subtract the prompt themselves.
- Output policy **pins every turn's length**: a replayed turn decodes
  exactly its recorded `completion_tokens`, a synthetic (`random`) turn
  decodes `--output-len`. Either way it is enforced with
  `max_completion_tokens` + `ignore_eos=true`, giving an exact,
  controllable decode workload. (A `natural` EOS mode was tried and
  removed — rationale under "Why output length is pinned" below.)

Frozen **profiles** bundle the selection knobs. They select *real*
sessions by peak context and never truncate turns, so each trajectory
keeps its natural length. `mixed` spans the whole usable band and is the
primary scan — one pool, no rebalancing, the way MLPerf runs a single
combined dataset. The published tiers and their pools:

| Tier | Peak context | Pool (sessions) | Turns (median) |
|---|---|---|---|
| `deepswe/mixed` | 8K–176K | 800 (fixed selection) | 49 |
| `deepswe/full` | no filter | 22,489 | 49 |
| `lmcache/full` | no filter | 767 | 8 |

`mixed` inherits the dataset's own distribution rather than a balanced
one: 20% of its sessions sit under 48K, 56% between 48K and 128K, and 25%
above 128K, with a median peak context of 89K. That is also why it is
expensive — a `mixed` plan at `C=32` (160 sessions) is 9,220 turns and
6.2M output tokens, several times the decode volume of the same
concurrency on the short sessions.

The band stops at 176K because 176K plus an 8K system prompt still leaves
roughly 18K of headroom under a 200K serving context window; sessions
above that cannot be replayed identically across models. The deepswe
tiers share `system_prompt_len = 8192`, since the system prompt is a
property of the agent rather than of the conversation's scale;
`lmcache/full` keeps its recorded system prompts instead.

The pool defaults to `2 * concurrency`, the sizing used throughout the
tool's cross-engine validation. Raise `--num-sessions` toward 5×C when
many *completed* sessions matter; pass the tier size for a full-tier
pass.

### 3. Fixed-concurrency long-tail → steady window, not truncation

Closed-loop runs with heterogeneous trajectory lengths suffer a
ramp/drain bias: at the edges the system is under-loaded, so turns
completing there look artificially fast.

We **do not** homogenize by truncation (that would distort the natural
workload and cannot lengthen short sessions). Instead:

1. **Replenishment absorbs heterogeneity.** Each of the `C` workers holds
   one session and pulls the next when it finishes, so a short trajectory
   just recycles faster and the active count stays pinned at `C`.
2. **Steady-state window.** Every turn is tagged with the active-session
   count at dispatch; headline metrics are computed only over turns with
   `active >= steady_frac * C` (default 0.9). `--duration` makes the pool
   loop and stops new dispatch at the deadline, eliminating drain.
3. **Concurrency audit.** The achieved-concurrency histogram is always
   printed so the trim is transparent.
4. **Per-bucket reporting.** E2E and TTFT are bucketed by turn index
   (T1 vs T2+) and by input-length band — never a single global mean —
   so short and long turns are compared apples-to-apples.

### 4. Decode-KV reuse → trace replay

The engine replays the *recorded* assistant text (`trace` is the only
content policy). This is byte-reproducible across models, but the assistant
slot differs from what the model actually generated, so the engine
re-prefills those tokens each turn and never reuses the decode-phase KV
across turns. This understates cache benefit — a deliberate trade for
reproducibility: content stays identical for every server under test.

> Distributed note: decode-KV reuse across turns additionally requires
> the session to land on the same worker each turn. Use `--routing-key`
> (attaches `X-SMG-Routing-Key = session_id`) so a sticky router
> preserves per-session cache locality.

## What the closed-loop engine adds over the open-loop path

Open-loop request-rate benchmarks (sglang's `bench_serving`, MLPerf's
Server scenario) drive traffic by arrival rate; Abench drives it by
active session count. The closed-loop engine is the increment this
benchmark contributes.

| Capability | Open-loop (request-rate) | Closed-loop (Abench) |
|---|---|---|
| Load model | request rate plus a concurrency *cap* | **exactly `C` active sessions**, refilled on completion |
| Per-turn output length | one representative (mean) length per conversation | **faithful per-turn** `completion_tokens` (trace policy) |
| Assistant text across turns | the model's own reply only | `trace` (byte-reproducible across models) |
| Ramp / drain bias | folded into the run averages | **steady-state window** (`--steady-frac`) plus concurrency histogram |
| Latency reporting | aggregate means | **T1 vs T2+** split and input-length buckets |
| Shared-prefix control | whatever the trace happens to carry | **`--system-prompt-len N`**, tiled to an exact token count, byte-identical across sessions |
| Session selection | count / offset | **`--min-turns` / `--max-turns`**, explicit **`--min-context` / `--max-context`** band |
| Cache / routing visibility | — | prefix **cache ratio**, **routing stickiness** |
| Pool exhaustion | n/a | **stop-on-exhaustion**: stop the clock when a freed slot cannot be refilled |

Use an open-loop runner for rate-based saturation curves; use Abench
when the question is per-turn cache behavior or faithful trajectory
replay at a held concurrency.

## Relation to MLPerf's agentic methodology

MLCommons' agentic-inference work for MLPerf Inference replays the same
DeepSWE trajectories, and its
[reference implementation](https://github.com/mlcommons/endpoints/tree/main/examples/10_Agentic_Inference)
independently arrived at the same execution shape as this benchmark:

| | MLPerf | here |
|---|---|---|
| Source of work | replay recorded trajectories, no live agent | same |
| Turn ordering | closed loop, one in-flight request per conversation | same |
| Load | fixed active conversations (`target_concurrency`) | `--max-concurrency` |
| Tool waits | explicit recorded `delay_seconds`, slept before each turn | same placement; the delay is extracted from agent-step timestamps (see `pre_gap_ms` above); `--no-gaps` disables |
| Session affinity | `X-Session-ID` header | `--routing-key` |
| Tail handling | stop measuring when the first slot has no next trajectory | stop-on-exhaustion |

Reporting follows it too: aggregate output tokens/s against per-user
output tokens/s, the latter excluding injected waits rather than charging
them to the server, so a run is a point on a Pareto curve rather than a
pass/fail verdict. Read the curve by its shape: points at or past
saturation have high run-to-run variance (queueing amplifies small
timing differences), so compare trends across concurrency levels rather
than the exact figures of an individual saturated point, and repeat runs
near the saturation boundary when a single number has to be quoted.

**Why output length is pinned (`trace`) rather than left to natural EOS.**
A `natural` mode — generation ends at the model's own EOS under a single
cap — was tried and removed. It only works when the model's own length on
these turns is close to what the trace recorded. MLPerf's published bands
(`[404,494]`, `[355,434]`) belong to their model-dataset pairs and do not
transfer. Measured on a large reasoning model with thinking enabled: every turn
ran to the 8192 cap, roughly 17x the recorded mean of 479, so the band
check could never pass and a single concurrency point would have taken
tens of hours. The tell is an achieved mean sitting at the cap, and a
per-turn time far above what the recorded lengths imply. The DeepSWE
trace was produced by a task-focused harness with a median of 178 output
tokens per turn; a thinking model given the same turns and no length pin
treats them as open-ended. Under `trace` on the same model the mean was
467.8 tokens per turn, inside MLPerf's Kimi band, so the decode volume is
of the intended magnitude even though the lengths are pinned rather than
emergent.

Two deliberate differences remain. Prompts are not salted, so cross-session
reuse of the shared system prefix is permitted here and the reported cache
ratio is correspondingly higher than a salted run would show. And there is
no accuracy scoring: this benchmark measures serving behaviour only, so it
cannot detect a server that returns fast garbage.

## Metrics

| Metric | Meaning |
|---|---|
| T1 TTFT | first-token latency of the first turn (cold prefill) |
| T2+ TTFT | first-token latency of later turns (prefix reuse) |
| TTFT degradation | `mean(T2+) / mean(T1)`; ~1.0 means cache is working |
| TPOT | mean inter-token latency |
| Server E2E (by turn / input-len bucket) | full server time per turn |
| Prefix cache ratio | server-reported `cached_tokens / input_tokens` |
| **Host cache ratio** | host(DRAM)-layer hits / input tokens; from `sglext.cached_tokens_details` (needs `return_cached_tokens_details`) |
| **Host share of cache** | host hits / all cache hits; >0 proves the L2 layer is serving, not just HBM |
| **KV pool utilization** | `used / capacity` from Prometheus `/metrics` (mean + p99) |
| **KV eviction delta** | total evictions during the run (COUNTER delta) |
| **KV queue depth** | `num_waiting_reqs` p99 from `/metrics` |
| Output / total throughput | tokens per second over the steady window |
| Trajectory inference time | sum of a session's per-turn E2E |
| Turn completion rate | successful turns / issued turns |
| Routing stickiness | fraction of sessions whose turns all hit one worker |

## Random mode (`--dataset-name random`)

When a dataset is unavailable or a controlled workload is needed, the
benchmark can generate random sessions with parameterized shape:

| Knob | Default | Effect |
|---|---|---|
| `--system-prompt-len N` | 8192 | cross-session shared prefix, byte-identical across all sessions (matches profile default) |
| `--turns` | 20 | Turns per session |
| `--input-increment` | 2048 | New context tokens added per turn |
| `--output-len` | 256 | Output tokens per turn |
| `--gap-policy` | `exponential:3000` | Inter-turn gap distribution |
| `--gap-cap-ms` | 30000 | Cap on any single gap |
| `--session-timeout` | 0 (off) | Per-session wall-clock limit (seconds) |

### Gap distribution — the key agentic parameter

Standard multi-turn benchmarks (sglang `multi_turn_chat`, mooncake
trace replay) assume zero inter-turn delay: each turn fires immediately
after the previous completes. Real agentic workloads have **tool waits**
between turns — the agent executes a tool, observes the output, then
continues. This gap directly affects:

- **Steady-state concurrency**: during gaps the server is under-loaded.
- **KV cache pressure**: longer gaps increase eviction risk.
- **Throughput attribution**: gaps are not charged to the server.

`--gap-policy` controls this distribution (all capped at `--gap-cap-ms`,
T1 always zero):

| Policy | Syntax | When to use |
|---|---|---|
| `exponential:mean` | right-skewed (default) | closest to real tool-execution time |
| `fixed:Ms` | constant | controlled variable experiments |
| `uniform:lo,hi` | even spread | sensitivity analysis |
| `zero` | no gap | degenerate to pure multi-turn chat |

### Per-session timeout

`--session-timeout S` caps the wall-clock time of a single session.

## Server-side KV pool scraping

When the server exposes Prometheus metrics (`--enable-metrics`), the
benchmark automatically scrapes `/metrics` every `--metrics-interval`
seconds (default 5) during the run. Six gauges/counters are collected:

| Metric | Type | What it shows |
|---|---|---|
| `cache_available_tokens` | GAUGE | free KV pool space |
| `cache_used_tokens` | GAUGE | occupied KV pool space |
| `cache_evictable_tokens` | GAUGE | evictable (radix tree) space |
| `num_evicted_requests` | COUNTER | cumulative evictions |
| `num_running_reqs` | GAUGE | in-flight requests |
| `num_waiting_reqs` | GAUGE | queued requests |

The scraper is **default-on** and **silently degrades**: if `/metrics`
returns non-200 or is unreachable, it prints one notice and skips — the
benchmark runs normally without server-side stats. This fills the gap
between client-observed `prefix_cache_ratio` (what hit) and
server-internal eviction/queue depth (why it missed).
