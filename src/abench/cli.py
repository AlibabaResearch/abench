# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Command-line interface: argument parsing and dispatch.

Usage examples
--------------

Prepare the default dataset once (deepswe/mixed, ~42 MB; lands under
``~/abench_data/<name>/<tier>/``)::

    python3 -m abench data deepswe/mixed

Run a single fixed-concurrency point against a served model::

    python3 -m abench run --model default \
        --dataset-name deepswe/mixed --max-concurrency 16
    # auto-detects localhost :30000 (sglang) / :8000 (vllm); pass
    # --port or --base-url only for non-default deployments

Smoke test with no download (locally generated random workload)::

    python3 -m abench run --dataset-name random \
        --turns 3 --output-len 64 --max-concurrency 4

What each command *does* lives in ``abench.runner``; this module only
builds the parser (layered help: ``--help`` core flags, ``--help-advanced``
everything) and dispatches.
"""

from __future__ import annotations

import argparse
import os
import sys

from abench.runner import cmd_data_entry, cmd_run

# ----------------------------------------------------------------------
# argument parsing
# ----------------------------------------------------------------------
# Options hidden from the default --help; shown via `<cmd> --help-advanced`.
# Layering only -- every hidden flag still works exactly as before.
_ADVANCED_DESTS = {
    "run": {
        "turns", "input_increment", "output_len", "gap_policy", "gap_cap_ms",
        "min_turns", "max_turns", "min_context", "max_context",
        "system_prompt_len", "top_k", "presence_penalty",
        "steady_frac", "warmup_requests",
        "duration", "session_timeout", "routing_key", "worker_header",
        "ready_check_timeout_sec", "max_duration",
        "disable_tqdm", "host", "port",
    },
    "data": set(),
}


# (cmd, dest) -> original help text, used by --help-advanced to un-hide.
_ADVANCED_ORIG: dict = {}


class _HelpAdvancedAction(argparse.Action):
    """Print the full help including advanced (hidden) options."""

    def __init__(self, option_strings, dest, cmd_name=None, **kwargs):
        kwargs["nargs"] = 0
        super().__init__(option_strings, dest, **kwargs)
        self._cmd_name = cmd_name

    def __call__(self, parser, namespace, values, option_string=None):
        restored = []
        for act in parser._actions:
            key = (self._cmd_name, act.dest)
            if key in _ADVANCED_ORIG:
                # the stored help may itself be None (flag had no help text)
                act.help = _ADVANCED_ORIG[key]
                restored.append(act)
        print(parser.format_help())
        for act in restored:
            act.help = argparse.SUPPRESS
        parser.exit()


def _layer_help(subparsers) -> None:
    """Suppress advanced options' help text on every subcommand."""
    for name, sp in subparsers.choices.items():
        for act in sp._actions:
            if act.dest in _ADVANCED_DESTS.get(name, set()):
                _ADVANCED_ORIG[(name, act.dest)] = act.help
                act.help = argparse.SUPPRESS
        sp.add_argument(
            "--help-advanced",
            action=_HelpAdvancedAction,
            cmd_name=name,
            help="Show ALL options, including advanced/internal ones.",
        )


class _Parser(argparse.ArgumentParser):
    """No prefix matching: `--mode` must not silently resolve to `--model`."""

    def __init__(self, *a, **kw):
        kw.setdefault("allow_abbrev", False)
        super().__init__(*a, **kw)


def build_parser() -> argparse.ArgumentParser:
    from abench._version import __version__

    # `python3 -m abench` leaves __main__.py in argv[0]; without an explicit
    # prog every usage line reads "usage: __main__.py run [-h]", which is not
    # something a reader can copy back into a shell.
    prog = (
        "python3 -m abench"
        if os.path.basename(sys.argv[0]) == "__main__.py"
        else None
    )

    parser = _Parser(
        prog=prog,
        description="ABench — closed-loop agentic inference benchmark "
                    "(replays recorded agent trajectories, or a synthetic "
                    "workload, against any OpenAI-compatible server)",
        epilog="Most users only need a few options per command; run "
               "`python3 -m abench <command> --help-advanced` to see everything.",
    )
    parser.add_argument("--version", action="version",
                        version=f"abench {__version__}")
    sub = parser.add_subparsers(
        dest="command", required=True, parser_class=_Parser
    )

    # data
    d = sub.add_parser(
        "data",
        help="Data preparation: download a preset dataset tier to the "
             "standard location <data-path>/<dataset>/<tier>/ "
             "(usage: data <DATASET>[/<TIER>], e.g. data deepswe/mixed).",
    )
    d.add_argument("dataset", nargs="?", default=None,
        help="DATASET[/TIER] to download, e.g. deepswe/mixed. "
             "A bare dataset name is rejected -- tiers are peers, no default. "
             "See --list for the available combinations.")
    d.add_argument("--data-path", default=os.path.expanduser("~/abench_data"),
        help="Download root; data lands under `<data-path>/<dataset>/<tier>/` "
             "-- the same layout `run --dataset-path` understands. "
             "Non-default roots: pass the printed --dataset-path back to run.")
    d.add_argument("--list", action="store_true",
        help="List the available dataset x tier combinations.")
    d.set_defaults(func=cmd_data_entry)

    # run
    r = sub.add_parser("run", help="Run the closed-loop replay benchmark.")
    r.add_argument("--base-url", default=None,
        help="Full server base URL, e.g. http://localhost:30000. When set it "
             "overrides --host/--port entirely.")
    r.add_argument("--host", default="0.0.0.0",
        help="Server host when --base-url is not given.")
    r.add_argument("--port", type=int, default=None,
        help="Server port. Unset: derived from --backend (sglang/openai "
             "30000, vllm 8000; auto probes 30000 then 8000). Pass only "
             "for non-default deployments.")
    r.add_argument("--model", default="default",
        help="Model name in the request body; a local directory is also "
             "auto-used as the tokenizer.")
    r.add_argument("--tokenizer", default=None,
        help="Explicit tokenizer path; auto-resolved when unset.")
    r.add_argument("--backend", choices=["auto", "sglang", "vllm", "openai"],
        default="auto",
        help="Serving backend. Recommended to leave unset: 'auto' probes "
             "the server (/server_info, then /metrics content) and gates "
             "the observation layer accordingly. A wrong explicit "
             "declaration does not error out, it silently drops that "
             "backend's observation metrics (cache breakdown / KV pool / "
             "server_info snapshot).")
    r.add_argument("--dataset-name", dest="dataset_name", default="deepswe/mixed",
        help="Dataset identity: '<name>/<tier>' (e.g. deepswe/mixed, the "
             "default) looked up under ~/abench_data, or 'random' for a "
             "locally generated workload. With --dataset-path it labels the "
             "run (the path supplies the location).")
    r.add_argument("--dataset-path", default=None,
        help="Data location: a .jsonl/.jsonl.zst file, a tier directory "
             "(holds manifest.json), or a root containing <name>/<tier>/. A "
             "manifest found at the target sets band/turn/system-prompt "
             "defaults like --dataset-name; without one, raw mode replays the "
             "files as-is (no band, min_turns 1, system_prompt_len 0).")

    # --- random workload knobs (only with --dataset-name random) ---
    r.add_argument("--turns", type=int, default=20,
        help="Turns per random session.")
    r.add_argument("--input-increment", type=int, default=2048,
        help="New context tokens added per turn (random mode).")
    r.add_argument("--output-len", type=int, default=256,
        help="Output tokens per turn (random mode).")
    r.add_argument("--gap-policy", default="exponential:3000",
        help="Inter-turn gap distribution: zero | fixed:Ms | uniform:lo,hi | exponential:mean.")
    r.add_argument("--gap-cap-ms", type=float, default=30_000.0,
        help="Cap on any single inter-turn gap (ms).")

    r.add_argument("--min-turns", type=int, default=None,
        help="Drop sessions with fewer turns than this (dataset manifest "
             "default applies when unset).")
    r.add_argument("--max-turns", type=int, default=None,
        help="Cap a session at this many turns; never pads short ones.")
    r.add_argument(
        "--min-context",
        type=int,
        default=None,
        help="Keep sessions whose replayed peak context (recorded peak + "
             "injected system prompt) is >= this many tokens. Same axis as "
             "--max-context.",
    )
    r.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="The MODEL context window the workload must fit: internally "
             "the recorded-peak cap becomes max_context minus the measured "
             "injected system prompt (the derivation is printed on the "
             "[plan] line).",
    )
    r.add_argument("--temperature", type=float, default=0.0,
        help="Sampling temperature (greedy by default, as in bench_serving).")
    r.add_argument("--top-p", type=float, default=1.0,
        help="Top-p; only sent when < 1.0.")
    r.add_argument(
        "--chat-template-kwargs",
        default=None,
        help='JSON forwarded as chat_template_kwargs, e.g. \'{"thinking": true}\' '
        "for reasoning models. Left unset the server template default applies.",
    )
    r.add_argument("--top-k", type=int, default=None,
        help="Top-k sampling; only sent when set.")
    r.add_argument("--presence-penalty", type=float, default=None,
        help="Presence penalty; only sent when set.")
    r.add_argument(
        "--header",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra HTTP header on every request, repeatable "
             "(e.g. --header X-Tenant=abc). Merged after the auth header, "
             "so it can override it.",
    )
    r.add_argument(
        "--extra-request-body",
        default=None,
        help="JSON object merged into the top level of every request body, "
        "overriding the built-in fields (e.g. '{\"seed\": 123}'). The "
        "escape hatch for engine-specific request parameters.",
    )
    r.add_argument("--system-prompt-len", type=int, default=None,
        help="Target token length of the shared synthetic system prompt "
             "(0 = none; dataset manifest default applies when unset). "
             "The actual injected length is measured and shown in the "
             "[plan] line, and added to the context band accounting.")
    r.add_argument("--num-sessions", type=int, default=None,
        help="Session pool size; defaults to 2 x --max-concurrency (the "
             "validated default sizing). Pass a larger count (or the "
             "tier size) for a full-tier pass.")
    r.add_argument("--seed", type=int, default=42,
        help="Sampling seed for session selection and random mode.")
    r.add_argument("--tag", default=None,
        help="Free-form label written into the result record; pure "
             "annotation, affects nothing (as in bench_serving).")

    r.add_argument("--max-concurrency", type=int, default=16,
        help="Simultaneously active agent sessions (closed loop: a finished "
             "session's slot is immediately refilled).")
    r.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds; enables looping pool + drain-free window.",
    )
    r.add_argument("--steady-frac", type=float, default=0.9,
        help="Steady window = turns dispatched at >= frac*C active "
             "sessions; excludes ramp/drain from metrics.")
    r.add_argument("--no-gaps", action="store_true", help="Ignore tool-wait gaps.")
    r.add_argument("--disable-tqdm", action="store_true",
        help="Disable the tqdm turn-progress bar (falls back to the "
             "periodic [progress] heartbeat; tqdm also absent from the "
             "environment does the same).")
    r.add_argument("--gap-scale", type=float, default=1.0,
        help="Scale factor applied to each tool-wait gap (pre_gap_ms) when "
        "gaps are respected: 0.5 halves every wait, 1.0 reproduces the "
        "recorded timing (default), 2.0 doubles it. Ignored with "
        "--no-gaps.")
    r.add_argument("--disable-stream", action="store_true",
        help="Non-streaming requests (no per-token ITL, whole-response "
             "E2E only).")
    r.add_argument("--flush-cache", action="store_true",
        help="POST /flush_cache (sglang) or /reset_prefix_cache (vllm; "
             "needs the server launched with VLLM_SERVER_DEV_MODE=1) "
             "before the run.")
    r.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Sessions replayed once before measurement to populate the "
             "prefix cache and surface config errors early (default 1). "
             "Each warmup session is capped to its first 3 turns and the "
             "phase shares the --max-duration fuse.",
    )
    r.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=60,
        help="Seconds to wait for GET /v1/models to answer before the run "
             "starts (0 = skip). Fails fast instead of timing out every "
             "request against a still-loading server.",
    )
    r.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Runaway fuse: hard cap (s) on total run wall clock. Default "
             "no cap; a healthy run should never reach it.",
    )
    r.add_argument(
        "--routing-key",
        action="store_true",
        help="Attach a stable X-SMG-Routing-Key per session.",
    )
    r.add_argument(
        "--worker-header",
        default=None,
        help=(
            "Response header carrying the serving worker's identity, used "
            "for the routing-stickiness metric (e.g. X-Worker-Id). Off by "
            "default: standard gateways do not return one."
        ),
    )
    r.add_argument(
        "--turn-timeout",
        type=float,
        default=3600.0,
        help="Per-request timeout in seconds; a hung turn frees its slot "
        "after this long (MLPerf's qwen config uses 14400).",
    )
    r.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between [progress] heartbeat lines (0 disables them).",
    )
    r.add_argument(
        "--metrics-interval",
        type=float,
        default=5.0,
        help="Seconds between /metrics scrapes for server-side KV pool stats "
        "(0 disables; silently skipped if server lacks --enable-metrics).",
    )
    r.add_argument(
        "--session-timeout",
        type=float,
        default=0.0,
        help="Per-session wall-clock timeout in seconds (0 = disabled). "
        "A session exceeding this is marked truncated.",
    )
    r.add_argument(
        "--output-details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the per-turn records (TTFTs, ITLs, errors) in the "
             "--output-file JSON. On by default -- a run you cannot "
             "re-examine afterwards is a run you have to repeat; pass "
             "--no-output-details to keep each line to config + metrics "
             "when the full per-turn record is not wanted.",
    )
    r.add_argument("--output-file", default=None,
        help="Append one JSON line per run (JSONL); per-turn records are "
             "included by default (--no-output-details opts out).")
    r.add_argument("--verbose", action="store_true",
        help="Verbose per-turn console output.")
    r.set_defaults(func=cmd_run)

    _layer_help(sub)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
