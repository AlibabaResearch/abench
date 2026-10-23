# Contributing to Abench

Abench is a closed-loop benchmark client. It holds a fixed number of
*agent sessions* active against an OpenAI-compatible inference server,
replays each session turn by turn with its recorded tool waits and decode
length, and reports per-turn latency, throughput, cache behaviour and
server-side KV pool statistics.

The contributions we most want are the ones that make those numbers more
trustworthy, or that widen the set of servers Abench can observe.

## Code of Conduct

This project follows the
[Alibaba Open Source Code of Conduct](https://github.com/AlibabaDR/community/blob/master/CODE_OF_CONDUCT.md).

## Contributor License Agreement (CLA)

Before we can merge your first pull request, you must sign the Abench
Contributor License Agreement (CLA). On your first PR the CLA Assistant
bot comments on the pull request with the CLA document and marks the
status check as failing. To sign, reply with exactly:

> I have read the CLA Document and I hereby sign the CLA

The check turns green after the signature is recorded. If you have
already signed, comment `recheck` to re-run the check. The workflow is
defined in `.github/workflows/cla.yml`; project maintainers and bots are
allowlisted and do not need to sign.

## Development environment

Python 3.9 or newer. No GPU and no inference server are needed to install
the package or to explore the CLI.

```bash
pip install -e '.[data,tokenizer,ui]'

python3 -m abench --version
python3 -m abench data --list          # every dataset x tier combination
python3 -m abench run --help           # core flags
python3 -m abench run --help-advanced  # everything
```

To exercise the replay path you need a server to point at:

```bash
# no download at all: a locally generated workload
python3 -m abench run --dataset-name random --turns 3 --output-len 64 \
    --max-concurrency 4

# or the real thing: ~42 MB tier from the ModelScope mirror, then replay it
python3 -m abench data deepswe/mixed
python3 -m abench run --dataset-name deepswe/mixed --max-concurrency 4
```

`--host`/`--port` are usually left unset: `auto` probes localhost :30000
(SGLang) then :8000 (vLLM).

## Layout

```
src/abench/
  cli.py          argument parsing and help layering only
  runner.py       what `data` and `run` actually do
  core/           domain: session data, workload planning, result records
  serve/          the server under test: endpoints, telemetry, report
  replay/         closed-loop engine and turn transport
docs/             reference (README.md) and design notes (DESIGN.md)
```

Dependencies point inward — `cli -> runner -> replay -> serve -> core`.
Please do not add an import that reverses one of those arrows, and do not
grow `cli.py` into orchestration: it parses arguments and dispatches.

## How to contribute

1. **Open an issue first** for anything that would change a reported
   number or add a flag. Measurement semantics are the product here, so
   they get discussed before they get implemented.
2. Fork the repository and branch from the default branch.
3. Keep one commit per concern. Say *why* in the message: a benchmark
   change is only reviewable if the reviewer can tell which measurement
   it affects.
4. Open a pull request. In the description, state what happens to the
   reported metrics — "no metric changes", "TTFT now excludes X", "new
   field `y`, existing fields untouched".
5. Make sure the CLA check passes (see above). Small fixes (typos, a
   wrong default in the docs, a clearer error message) can go straight
   to a pull request.

## What a change must not do

- **Silently change what a metric means.** If a definition has to move,
  move it in `docs/README.md` in the same pull request and say so in the
  description. Existing result files are compared against new ones.
- **Let the model's own EOS decide a turn's length.** Every output policy
  pins a length and the engine sends `ignore_eos`; that is what makes two
  engines comparable on the same workload. A reasoning model left to
  itself runs to the generation cap and emits several times the recorded
  work.
- **Report an unsupported metric as `0`.** It is `null`. "Not measured"
  and "measured zero" are different facts, and collapsing them has
  already produced a false comparison once.
- **Add a dependency that is not on PyPI**, or make an existing optional
  extra mandatory. The runtime dependency is `aiohttp`; `zstandard`,
  `transformers`/`tiktoken` and `tqdm` all have working fallbacks.
- **Document a flag that does not exist, or leave a new flag
  undocumented.** `docs/README.md` aims to cover every flag; the check is
  a set difference between `--help-advanced` output and the flag table.

## Verifying a change

This repository does not distribute a test suite. Before opening a pull
request, please confirm at minimum:

- `python3 -m abench --version`, `data --list` and both `--help` layers
  still work;
- a `--dataset-name random` run against a server completes and prints a
  report;
- if you touched the observation layer, a run against the affected
  backend still reports its metrics (and still reports `null`, not `0`,
  on a backend that lacks them);
- `python -m build --wheel` succeeds and the wheel contains the package
  plus `abench/data/recording_rates.json`.

## License

Abench is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). By
submitting a contribution you agree to license it under the same terms.
