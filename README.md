# Abench

English | [简体中文](README.zh.md)

Agentic inference benchmark: closed-loop replay of recorded agent
trajectories (**DeepSWE v1.1**, **LMCache**) — or a locally generated
workload — against any OpenAI-compatible inference server
(SGLang, vLLM, ...).

It answers: **how fast does a single agent advance?** — per-turn
TTFT / TPOT / E2E, cache behavior, and throughput at a fixed concurrency,
on a fixed, reproducible agent workload.

## Install

```bash
pip install agentic-bench          # from PyPI (after the first release)
```

Or install the wheel attached to a GitHub Release:

```bash
pip install https://github.com/AlibabaResearch/abench/releases/download/v0.2.0/agentic_bench-0.2.0-py3-none-any.whl
```

From a checkout:

```bash
pip install .
```

The PyPI distribution is named `agentic-bench`; the Python module and the
`abench` command keep the project name. An unrelated PyPI package named
`abench` (ActivitySim) also installs an `abench` module and command, so do
not install both in the same environment. If you already have it, run
`pip uninstall abench` first.

## Quick start

Two subcommands.

```bash
# 1) one-time data prep: downloads deepswe/mixed (~42 MB, SHA-256
#    verified, resumable) from the ABench ModelScope mirror; data lands
#    under ~/abench_data/deepswe/mixed/
python3 -m abench data deepswe/mixed

# 2) replay at one concurrency; every session runs to completion
#    (or under --duration) and full metrics are reported. --output-file
#    appends one JSON line per run; per-turn records are included by
#    default (--no-output-details keeps lines slim).
python3 -m abench run \
    --model default \
    --dataset-name deepswe/mixed --max-concurrency 16 \
    --output-file results.jsonl
# no --host/--port needed: localhost :30000 (sglang) / :8000 (vllm) is
# auto-detected; pass --port/--base-url only for non-default deployments
```

Smoke test with no download (locally generated random workload):

```bash
python3 -m abench run \
    --dataset-name random \
    --turns 3 --output-len 64 --max-concurrency 4
```

`python3 -m abench ...` is the primary invocation (matching how
`bench_serving` is called). The wheel also installs an equivalent
`abench` console script. Results always land on disk: without
`--output-file` the run writes
`{backend}_{date}_{dataset}.jsonl` in the working directory.

## Backends & auth

Any OpenAI-compatible server works; `--backend auto|sglang|vllm|openai`
(default `auto`, probed from the server) adapts the *observation* layer —
SGLang-only extras (per-layer cache breakdown, `/flush_cache`, `/metrics`
gauge names) are gated per backend, and unsupported metrics report `null`
rather than `0`. Request shape is standard OpenAI either way.

**Leave `--backend` and `--port` unset**: `auto` probes the engine and
resolves the port (sglang/openai 30000, vLLM 8000). A wrong explicit
`--backend` does not error — it silently drops that backend's metrics
(cache breakdown / KV pool / server_info); `abench` warns when a declared
backend does not match the server.

- **Auth**: `export OPENAI_API_KEY=...` (or `API_KEY=` for custom gateway
  schemes) — attached to every request, no CLI flag needed.
- **Escape hatches**: `--header Key=Value` (repeatable) for custom HTTP
  headers; `--extra-request-body '{"seed": 123}'` merges into every
  request body and can override built-in fields.
- `--ready-check-timeout-sec` polls `/v1/models` before the run so a
  still-loading server fails fast instead of timing out every request.

## CLI

| command | purpose |
|---|---|
| `python3 -m abench data <DATASET>/<TIER>` | the single data-prep entry: download a preset dataset tier (SHA-256 verified, resumable) from the [ABench ModelScope mirror](https://www.modelscope.cn/datasets/Apsara-Stack/ABench) to `<data-path>/<dataset>/<tier>/` (`--list` shows all; a bare dataset name without a tier is rejected) |
| `python3 -m abench run` | the single bench entry: closed-loop replay at one concurrency; data located by `--dataset-name` (identity, looked up under `~/abench_data`) and/or `--dataset-path` (file / tier dir / root) |

`--help` shows the core flags; `--help-advanced` shows everything. The
tokenizer resolves automatically (`--tokenizer`, else a local `--model`
dir, else the server's `/model_info`, else `AutoTokenizer.from_pretrained`
for non-local names, else `tiktoken`, else chars/4) — the Hub fetch
matches `bench_serving`; set `HF_HUB_OFFLINE=1` on air-gapped machines.

## Docs

- [docs/README.md](docs/README.md) — reference: closed-loop protocol,
  dataset tiers, every flag, metric definitions
- [docs/DESIGN.md](docs/DESIGN.md) — design rationale, measurement decisions
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute, CLA, and review expectations
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Development

```
src/abench/
  cli.py          argument parsing and dispatch (python3 -m abench)
  runner.py       what `data` and `run` actually do
  core/           domain: session data, workload planning, result records
  serve/          the server under test: endpoints, telemetry, report
  replay/         closed-loop engine and turn transport
docs/             reference and design notes
```

```bash
pip install -e .
python -m build --wheel   # ships the package only
```

## License

Copyright (c) 2026, Alibaba Group and its affiliates. Abench is licensed
under the Apache License, Version 2.0 — see [LICENSE](LICENSE). The
upstream conventions this CLI follows, and where the datasets come from,
are recorded in [NOTICE](NOTICE); the repository itself ships no
trajectory data.
