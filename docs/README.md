# Abench reference: protocol, datasets, flags, metrics

Detailed reference for Abench, the closed-loop agentic inference benchmark —
see the [root README](../README.md) for install and quick start.

Closed-loop replay of real **DeepSWE v1.1** agent trajectories against an
OpenAI-compatible inference endpoint. It answers: **how fast does a single
agent advance?** — per-turn TTFT / TPOT / E2E, cache behavior, and
throughput at a fixed concurrency.

> Targets SGLang / vLLM style servers: the replay relies on `ignore_eos`
> and `max_completion_tokens` to pin exact decode lengths, which strict
> OpenAI implementations may reject.

This is an *Agentic Inference Performance* benchmark — it measures serving
speed on a fixed, reproducible agent workload. It does **not** score tool
correctness or task success.

See [`DESIGN.md`](./DESIGN.md) for the full rationale.

## Closed-loop model

Abench replays each recorded agent session as a live conversation: a fixed
number of concurrent agents (active sessions, replenished on completion),
real tool-wait gaps, pinned per-turn decode lengths, and per-turn latency
buckets (T1 cold start vs T2+ prefix reuse) — plus steady-state windowing
and pinned trace replay.

## 1. Data preparation (`data`, one command)

Datasets are distributed from the ABench mirror on ModelScope
([Apsara-Stack/ABench](https://www.modelscope.cn/datasets/Apsara-Stack/ABench))
as Zstandard-compressed JSONL (one line = one session, format
`abench-session-jsonl-v1`). Every download is SHA-256 verified and
resumable; re-running the same command is a no-op cache hit.

```bash
# the main scan tier: deepswe/mixed, ~42 MB, lands under
# <data-path>/<dataset>/<tier>/ (default ~/abench_data/deepswe/mixed/)
python3 -m abench data deepswe/mixed

# the LMCache agent traces (~9 MB, 767 sessions)
python3 -m abench data lmcache/full

# the whole DeepSWE v1.1 corpus (~1.8 GB, 22,489 sessions, 4 parts)
python3 -m abench data deepswe/full

# what's available
python3 -m abench data --list
```

Each tier lands as `*.jsonl.zst` plus a `manifest.json` (band and
system-prompt knobs, file checksums) under `<data-path>/<dataset>/<tier>/`.
The tier catalog is hardcoded in `abench.core.dataset.TIERS`; it changes only
when a new version is published to the mirror.

`data` takes two flags: `--data-path DIR` sets the download root (default
`~/abench_data`) and `--list` prints every dataset × tier combination. The
layout under `--data-path` is the one `run --dataset-path` reads, so with a
non-default root you pass the printed directory back to `run`.

`run` loads every session in the dataset directory, then applies the
manifest's context band (for tiers that define one) plus any explicit
`--min-context/--max-context/--min-turns` filters.

## 2. Run (closed-loop replay)

```bash
python3 -m abench run \
    --model default \
    --dataset-name deepswe/mixed --max-concurrency 16
# localhost :30000 (sglang) / :8000 (vllm) is auto-detected; pass
# --port/--base-url only for non-default deployments
```

Every session is played exactly once (or, with `--duration`, the pool
loops and dispatch stops at the deadline). Full metrics are reported on
the console; `--output-file` appends one JSON line per run (JSONL), with
per-turn records included by default (`--no-output-details` keeps lines
slim). Results always land on disk: without `--output-file` the run writes
`{backend}_{date}_{dataset}.jsonl` in the working directory, and `--tag`
labels the record for telling rows apart. On the sglang backend the
result line also embeds a `/server_info` snapshot, and
`--ready-check-timeout-sec` polls `/v1/models` before the run so a
still-loading server fails fast. Progress shows as a tqdm turn bar when
tqdm is installed (heartbeat lines otherwise).

### Tokenizer resolution (no `--tokenizer` needed in most setups)

Token counting uses, in order:

1. `--tokenizer <path-or-name>` if given;
2. `--model` when it is a local directory containing tokenizer files;
3. the running server's `/model_info` endpoint (`tokenizer_path` /
   `model_path` fields — works when bench and server share a filesystem);
4. a non-local name handed to `AutoTokenizer.from_pretrained` (an HF repo
   id, fetched from the Hub — same resolution as sglang's `bench_serving`);
5. tiktoken `cl100k_base`, then chars/4, as the offline fallback.

Step 4 can reach the HuggingFace Hub (like `bench_serving`); on an
air-gapped machine pass a local `--model`/`--tokenizer` directory, or set
`HF_HUB_OFFLINE=1` / `HF_ENDPOINT=https://hf-mirror.com`. If the Hub fetch
fails, steps 5 keep the run going instead of hard-erroring.

### Backends & auth

`--backend auto|sglang|vllm|openai` (default `auto`: `/server_info`
answering means sglang, a `/metrics` body containing `vllm:` means vLLM,
anything else falls back to the generic OpenAI profile) selects the
observation layer: SGLang-only extras (`return_cached_tokens_details`,
the per-layer cache breakdown, `/flush_cache`, `/model_info`, `sglang:`
gauge names) are sent or read only for sglang; vLLM maps its own gauges
(`num_requests_running` / `num_requests_waiting`,
`gpu_cache_usage_perc`); the generic profile sends plain requests and
skips `/metrics`. Metrics a backend cannot provide are reported as
`null`, not `0`.

Authentication is environment-based, as in sglang's `bench_serving`:
`OPENAI_API_KEY` is sent as a Bearer token; otherwise `API_KEY` is sent
verbatim (custom gateway schemes); neither set sends no auth header.
`--header Key=Value` adds arbitrary headers on every request.

Synthetic mode (no dataset required):

```bash
python3 -m abench run \
    --model default \
    --dataset-name random --turns 20 --max-concurrency 16 \
    --gap-policy exponential:3000 --session-timeout 600
```

## Datasets & tiers

Tiers select real sessions by their **peak recorded context** and never
truncate turns, so each trajectory keeps its natural length.

| Tier | Peak context band | Sessions | Typical turns | `system_prompt_len` |
|---|---|---|---|---|
| `deepswe/mixed` | 8K–176K | 800 | ~49 | 8192 |
| `deepswe/full` | no filter | 22,489 | ~49 | 8192 |
| `lmcache/full` | no filter | 767 | ~8 | 0 (recorded prompt) |

`mixed` keeps the DeepSWE dataset's own distribution (20% under 48K, 56%
48-128K, 25% above 128K). Expect it to be several times heavier than the
short sessions at the same concurrency. The band is capped at 176K so
that 176K plus an 8K system prompt still leaves headroom under a 200K
serving context window. `lmcache` replays the LMCache agentic traces
with their recorded system prompts and output lengths.

## Key flags

| Flag | Meaning |
|---|---|
| `--dataset-name <name>[/<tier>]` | dataset identity, `deepswe/mixed` (default); looked up under `~/abench_data/<name>/<tier>/`, or `random` for a locally generated workload |
| `--dataset-path PATH` | data location: a `.jsonl`/`.jsonl.zst` file, a tier directory (holds `manifest.json`), or a root containing `<name>/<tier>/`. A manifest sets tier-mode defaults; without one, raw mode (no band, min_turns 1, system_prompt_len 0) |
| `--backend {auto,sglang,vllm,openai}` | which server flavor is behind `--base-url`; `auto` probes. Gates the observation layer only — request shape is standard OpenAI either way |
| `--system-prompt-len N` | target token length of the shared synthetic system prompt (0 = none); the measured length is shown in the `[plan]` line |
| `--min-turns` / `--max-turns` | filter short sessions / cap long ones (never pads) |
| `--min-context` / `--max-context` | keep sessions whose replayed peak context (recorded peak + injected system prompt) falls in this token band (inclusive); the tier manifest's band applies when the CLI flags are unset. `--max-context` is the model's context window -- the tool derives the recorded-peak cap (`max_context - system prompt`) for you and prints it on the `[plan]` line |
| `--temperature` / `--top-p` / `--top-k` / `--presence-penalty` | sampling sent to the server; greedy by default (0.0 / 1.0, matching sglang's bench_serving; top_p only sent when < 1.0) |
| `--chat-template-kwargs '<json>'` | JSON forwarded as `chat_template_kwargs`, e.g. `'{"thinking": true}'` for a reasoning model. Unset, the server's template default applies |
| `--max-concurrency C` | number of simultaneously active agent sessions |
| `--tag T` | free-form label written into the result record (pure annotation, as in bench_serving) |
| `--num-sessions N` | pool size; defaults to 2 × `--max-concurrency` (the validated comparison protocol — override with the tier size for a full-tier pass) |
| `--seed N` | sampling seed for session selection and for the synthetic workload (default 42); two runs at the same seed pick the same sessions |
| `--duration S` | optional: loop the pool and stop at S seconds; without this, **default is stop-on-exhaustion** (when a session slot can't be filled from the pool, all sessions stop at the next turn boundary) |
| `--steady-frac` | steady window = turns with `active >= frac*C` (default 0.9); excludes ramp/drain from metrics |
| `--routing-key` | attach `X-SMG-Routing-Key = session_id` for sticky routers |
| `--worker-header` | response header carrying worker identity (e.g. `X-Worker-Id`); required for the routing-stickiness metric, off by default. No stock gateway returns one (`sgl-model-gateway` included), so stickiness stays empty unless your router is patched to emit it |
| `--turns N` | turns per random session (default 20) |
| `--input-increment K` | new context tokens added per turn in random mode (default 2048) |
| `--output-len K` | output tokens per turn in random mode (default 256) |
| `--gap-policy` | `zero` \| `fixed:Ms` \| `uniform:lo,hi` \| `exponential:mean` (default `exponential:3000`); inter-turn tool-wait distribution, T1 always zero |
| `--gap-cap-ms` | cap on any single inter-turn gap (default 30000) |
| `--gap-scale F` | multiplier on every replayed tool-wait gap (default 1.0 = recorded timing); 0.5 halves each wait, 2.0 doubles. Ignored with `--no-gaps` |
| `--session-timeout S` | per-session wall-clock limit in seconds (0 = disabled); a session exceeding this is marked truncated |
| `--turn-timeout S` | per-request timeout in seconds (default 3600); a hung turn gives up its slot after this long instead of holding the run open. Raise it for very long generations |
| `--metrics-interval S` | seconds between `/metrics` scrapes for server-side KV pool stats (default 5; 0 disables; silently skipped if server lacks `--enable-metrics`) |
| `--flush-cache` | drop the server's prefix cache before the run: `POST /flush_cache` on sglang, `POST /reset_prefix_cache` on vLLM (the server must be launched with `VLLM_SERVER_DEV_MODE=1`; a 404 is reported and the run continues), skipped on openai. Use it when comparing back to back — a cache left warm by the previous run inflates the next one |
| `--header Key=Value` | extra HTTP header on every request, repeatable |
| `--extra-request-body '<json>'` | JSON object merged into every request body (overrides built-in fields) — the escape hatch for engine-specific parameters |
| `--warmup-requests N` | sessions replayed once before measurement (default 1) |
| `--ready-check-timeout-sec S` | wait for `GET /v1/models` before the run (default 60; 0 skips) |
| `--max-duration S` | runaway fuse: hard wall-clock cap on the whole run |
| `--output-details` / `--no-output-details` | per-turn records in `--output-file` (JSONL, one line per run); **on by default** |
| `--disable-stream` | send non-streaming requests. Costs the per-token ITL distribution and leaves whole-response E2E only, so leave it off unless the server cannot stream |
| `--disable-tqdm` | turn off the tqdm turn-progress bar and fall back to the `[progress]` heartbeat (also what happens when tqdm is not installed) |
| `--progress-interval S` | seconds between `[progress]` heartbeat lines (default 30; 0 disables them) |
| `--verbose` | per-turn console output, on top of the progress line |
| `--version` | top-level: print the version and exit (`python3 -m abench --version`) |

> **Check the model's context length against the profile band.** The
> standard profiles reach up to 176K peak context. If the served model's
> native context is smaller (e.g. a 32K model), every session that
> outgrows it is rejected with HTTP 400 forever after — the failure is
> silent (visible only via the failed-turn count in the report) and the
> latency numbers are computed over a shrinking subset. Cap the band to
> the model's limit with
> `--min-context 8000 --max-context <model_ctx>` -- the tool
> subtracts the injected system prompt itself (see the `[plan]` line).

> **Pool sizing note (stop-on-exhaustion):** the pool defaults to
> `2 × --max-concurrency` — a sizing validated across engines and
> models. With a small pool the drain phase truncates up to
> `C-1` in-flight sessions (reported as `truncated`, excluded from
> session-level stats — not failures), so raise `--num-sessions` (e.g.
> 5×C) or use `--duration` (looping pool) when you need many *completed*
> sessions for `completed sessions` / `mean trajectory time`. Per-turn
> metrics are unaffected — they use the steady-state window.

## Metrics

Reported over the steady-state window, bucketed by turn index and
input-length band: **T1 / T2+ TTFT**, **TTFT degradation**, **TPOT**,
**ITL distribution (p50/p90/p95/p99/max)**, **Server E2E**, **prefix
cache ratio** (server-reported), **output/total throughput**,
**trajectory inference time**, **turn completion rate**, and the
**achieved-concurrency histogram** (for auditability).

When the server exposes `/metrics` (`--enable-metrics`), the benchmark
also scrapes and reports **KV pool utilization** (mean + p99), **eviction
delta**, and **queue depth** (p99). See `DESIGN.md` for the full list.

