# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Command orchestration: what `abench data` and `abench run` actually do.

Resolves the dataset source and sampling knobs, plans the workload, drives
warmup and the measured closed-loop run, then hands results to the report
layer. Argument *parsing* lives in ``abench.cli``; talking to the server
lives in ``abench.serve``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Optional

from abench import (
    PROFILES,
    build_system_prompt,
    compute_metrics,
    download_tier,
    fetch_server_info,
    flush_cache,
    load_sessions,
    iter_jsonl_sessions,
    measure_text_tokens,
    plan_workload,
    print_report,
    run_closed_loop,
    save_results,
    select_sessions,
    summarize_server_metrics,
    synthesize_sessions,
    warmup,
    wait_for_server_ready,
    write_tier_manifest,
)
from abench.core.dataset import TIERS, NormSession
from abench.serve.endpoints import (
    _detect_backend,
    _fetch_max_model_len,
    _parse_extra_request_body,
    _parse_header_args,
    _resolve_api_url,
    _resolve_base_url,
    _verify_declared_backend,
    load_tokenizer,
)

# ----------------------------------------------------------------------
# data subcommand (ModelScope download)
# ----------------------------------------------------------------------
def cmd_data_entry(args: argparse.Namespace) -> None:
    """`python3 -m abench data`: download a dataset x tier from ModelScope.

    Data lands under ``<data-path>/<dataset>/<tier>/`` as ``*.jsonl.zst``
    plus a ``manifest.json`` carrying the tier's selection knobs.
    """
    if args.list:
        _list_datasets()
        return
    spec_str = (args.dataset or "").strip().strip("/")
    if not spec_str:
        print(
            "usage: python3 -m abench data <DATASET>[/<TIER>]\n"
            "       e.g. python3 -m abench data deepswe/mixed\n"
            "       see --list for the available combinations",
            file=sys.stderr,
        )
        sys.exit(1)
    if "/" in spec_str:
        dataset, tier = (p.strip() for p in spec_str.split("/", 1))
    else:
        dataset, tier = spec_str, ""
    if dataset == "random":
        print("random needs no download: python3 -m abench run --dataset-name random", file=sys.stderr)
        sys.exit(1)
    if not tier:
        print(
            "error: specify a tier (see --list): "
            + " / ".join(TIERS),
            file=sys.stderr,
        )
        sys.exit(1)
    key = f"{dataset}/{tier}"
    spec = TIERS.get(key)
    if spec is None:
        print(
            f"unknown dataset/tier {key!r}; see `python3 -m abench data --list`",
            file=sys.stderr,
        )
        sys.exit(1)
    out_dir = os.path.join(args.data_path, dataset, tier)
    files = download_tier(spec, out_dir, verbose=True)
    write_tier_manifest(spec, out_dir, files)
    total_mb = sum(os.path.getsize(p) for p in files) / 1e6
    print(
        f"[data] {key}: {spec.sessions} sessions, {len(files)} file(s), "
        f"{total_mb:.1f} MB -> {out_dir}"
    )
    if os.path.abspath(os.path.expanduser(args.data_path)) == DEFAULT_DATA_ROOT:
        print(f"[data] run with: python3 -m abench run --dataset-name {key}")
    else:
        print(
            f"[data] run with: python3 -m abench run --dataset-name {key} "
            f"--dataset-path {out_dir}"
        )


def _split_dataset_spec(spec: str) -> tuple:
    """Split a `<dataset>/<tier>` spec into (dataset, tier).

    A bare dataset name defaults to the 'mixed' tier.
    """
    name = (spec or "deepswe/mixed").strip()
    if "/" in name:
        ds, _, tier = name.partition("/")
        return (ds or "deepswe"), (tier or "mixed")
    return name, "mixed"


# Fixed root for the standard layout written by `python3 -m abench data`.
DEFAULT_DATA_ROOT = os.path.expanduser("~/abench_data")


def _raw_knobs(args: argparse.Namespace) -> dict:
    """Knobs for raw data (no manifest): no band filter, min_turns floor
    of 1, no synthesized system prompt. Explicit CLI knobs still win."""
    return dict(
        min_turns=args.min_turns if args.min_turns is not None else 1,
        max_turns=args.max_turns,
        min_context=args.min_context,
        max_context=args.max_context,
        output_policy="trace",
        system_prompt_len=args.system_prompt_len
        if args.system_prompt_len is not None
        else 0,
    )


def _pool_size(args: argparse.Namespace) -> int:
    """Pool size for both branches: --num-sessions when given, else the
    validated 2 x max-concurrency default.
    Uses an explicit None check so a passed 0 is not silently replaced."""
    if args.num_sessions is not None:
        return args.num_sessions
    size = 2 * max(1, args.max_concurrency)
    print(
        f"[plan] num-sessions unset -> pool = 2 x C = {size} sessions "
        "(pass --num-sessions to override; see the pool-sizing note)"
    )
    return size


def _resolve_data_source(
    dataset_path: str, ds_name: str, tier: str
) -> tuple:
    """Resolve --dataset-path to (single_file, data_dir, manifest_dir).

    Search order: file (use its dir's manifest if present) -> this dir
    holds manifest.json (tier dir) -> descend one level to
    ``<name>/<tier>/manifest.json`` (root dir) -> bare data dir (raw).
    ``manifest_dir`` is None for raw mode; ``data_dir`` is where the
    session files live. Caller treats a nonexistent path as raw with an
    empty load.
    """
    p = os.path.abspath(os.path.expanduser(dataset_path))
    if os.path.isfile(p):
        parent = os.path.dirname(p)
        man = parent if os.path.exists(os.path.join(parent, "manifest.json")) else None
        return p, parent, man
    cand = os.path.join(p, ds_name, tier)
    if os.path.exists(os.path.join(p, "manifest.json")):
        return None, p, p
    if os.path.exists(os.path.join(cand, "manifest.json")):
        return None, cand, cand
    return None, p, None


def _dataset_dir(spec: str, data_path: str) -> str:
    """Standard on-disk location for a `<dataset>/<tier>` spec:
    ``<data-path>/<dataset>/<tier>/``, as written by
    ``python3 -m abench data``."""
    ds, tier = _split_dataset_spec(spec)
    return os.path.join(data_path, ds, tier)


def _resolve_profile_knobs(
    args: argparse.Namespace, data_dir: Optional[str] = None
) -> dict:
    """Fill in any knob the user did not set explicitly.

    Band defaults come from the dataset directory's ``manifest.json``
    (written by ``python3 -m abench data``): a ``band`` entry selects by peak context,
    an absent/null band means no filtering (e.g. full / lmcache tiers).
    Only a directory *without* a manifest falls back to the
    ``PROFILES['mixed']`` preset. Explicit CLI knobs always win.
    """
    defaults: dict = {}
    have_manifest = False
    if data_dir:
        try:
            with open(os.path.join(data_dir, "manifest.json")) as f:
                m = json.load(f)
            have_manifest = True
        except (OSError, ValueError):
            m = {}
        if not isinstance(m, dict):
            m = {}
        band = m.get("band")
        if isinstance(band, (list, tuple)) and len(band) == 2:
            defaults["min_context"] = band[0]
            defaults["max_context"] = band[1]
        for k in ("min_context", "max_context", "min_turns", "system_prompt_len"):
            v = m.get(k)
            if v is not None:
                defaults[k] = v
    if not have_manifest:
        p = PROFILES.get("mixed")
        if p is not None:
            for k in ("min_turns", "min_context", "max_context", "system_prompt_len"):
                defaults.setdefault(k, getattr(p, k, None))
    # A manifest without a band explicitly means "no context filtering";
    # leaving it unset would re-apply the mixed band and drop everything.
    if have_manifest and "min_context" not in defaults:
        defaults["min_context"] = None
        defaults["max_context"] = None
    knobs = dict(
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        min_context=args.min_context,
        max_context=args.max_context,
        output_policy="trace",
        system_prompt_len=args.system_prompt_len,
    )
    for k in ("min_turns", "min_context", "max_context", "system_prompt_len"):
        if knobs[k] is None:
            knobs[k] = defaults.get(k)
    # final fallbacks
    if knobs["min_turns"] is None:
        knobs["min_turns"] = 1
    if knobs["system_prompt_len"] is None:
        knobs["system_prompt_len"] = 0
    return knobs


def _prepare_plan(args: argparse.Namespace, tokenizer) -> tuple:
    """Build the workload plan; returns (plan, knobs, sampling, force_lengths)."""
    routing_fn = None
    if getattr(args, "routing_key", False):
        routing_fn = lambda s: s  # noqa: E731 - stable per-session key

    if args.dataset_name == "random":
        knobs = _resolve_profile_knobs(args)
        try:
            plan = synthesize_sessions(
                num_sessions=_pool_size(args),
                turns=args.turns,
                input_increment=args.input_increment,
                output_len=args.output_len,
                system_prompt_len=(
                    args.system_prompt_len
                    if args.system_prompt_len is not None
                    else 8192
                ),
                gap_policy=args.gap_policy,
                gap_cap_ms=args.gap_cap_ms,
                tokenizer=tokenizer,
                routing_key_fn=routing_fn,
                seed=args.seed,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        if not plan:
            print("no sessions synthesized; nothing to run", file=sys.stderr)
            sys.exit(1)
        knobs["system_prompt_len_actual"] = measure_text_tokens(
            plan[0].system_prompt, tokenizer
        )
        knobs["output_policy"] = f"fixed:{args.output_len}"
        print(
            f"[plan] random: {len(plan)} sessions x {args.turns} turns, "
            f"input_increment={args.input_increment}, output_len={args.output_len}, "
            f"gap_policy={args.gap_policy}"
        )
    else:
        ds_name, tier = _split_dataset_spec(args.dataset_name)
        if args.dataset_path:
            single_file, data_dir, manifest_dir = _resolve_data_source(
                args.dataset_path, ds_name, tier
            )
            if manifest_dir:
                knobs = _resolve_profile_knobs(args, manifest_dir)
                print(
                    f"[plan] dataset-path: applying manifest.json from "
                    f"{manifest_dir} (band/turn/system-prompt defaults active)"
                )
            else:
                knobs = _raw_knobs(args)
                print("[plan] no manifest found; running raw (no band/prompt defaults)")
            if single_file:
                sessions = [
                    NormSession.from_dict(raw)
                    for raw in iter_jsonl_sessions([single_file])
                ]
            else:
                sessions = load_sessions(data_dir)
        else:
            data_dir = _dataset_dir(args.dataset_name, DEFAULT_DATA_ROOT)
            if not os.path.isdir(data_dir):
                print(
                    f"no dataset directory {data_dir}; run "
                    f"`python3 -m abench data {ds_name}/{tier}` first, "
                    "or point --dataset-path at the data",
                    file=sys.stderr,
                )
                sys.exit(1)
            knobs = _resolve_profile_knobs(args, data_dir)
            sessions = load_sessions(data_dir)
        if not sessions:
            print(
                f"no sessions in {args.dataset_path or data_dir}; check the "
                f"path or run `python3 -m abench data {ds_name}/{tier}`",
                file=sys.stderr,
            )
            sys.exit(1)
        # The shared system prompt is built once up front so its measured
        # length can participate in the context band: the band must
        # describe what the server actually sees (recorded peak + injected
        # prompt), not the recorded peak alone.
        system_prompt = (
            build_system_prompt(knobs["system_prompt_len"], tokenizer)
            if knobs["system_prompt_len"]
            else ""
        )
        sys_len = measure_text_tokens(system_prompt, tokenizer)
        knobs["system_prompt_len_actual"] = sys_len
        # --max-context is the MODEL context window (user-facing contract):
        # the injected system prompt consumes part of it, so the recorded-
        # peak cap the band enforces is max_context - sys_len. Surface the
        # derivation instead of making the user subtract.
        if (
            knobs["max_context"] is not None
            and knobs["max_context"] <= sys_len
        ):
            print(
                f"error: --max-context {knobs['max_context']} leaves no room "
                f"for the {sys_len}-token injected system prompt "
                f"(recorded-peak cap would be <= 0)",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            selected = select_sessions(
                sessions,
                min_turns=knobs["min_turns"],
                max_turns=knobs["max_turns"],
                min_context=knobs["min_context"],
                max_context=knobs["max_context"],
                num_sessions=_pool_size(args),
                seed=args.seed,
                sys_prompt_tokens=sys_len,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        if not selected:
            print("no sessions matched the selection knobs", file=sys.stderr)
            sys.exit(1)
        plan = plan_workload(
            sessions=selected,
            system_prompt_len=knobs["system_prompt_len"],
            output_policy=knobs["output_policy"],
            tokenizer=tokenizer,
            routing_key_fn=(lambda s: s.session_id) if args.routing_key else None,
            system_prompt=system_prompt,
        )
        spl_actual = knobs.get("system_prompt_len_actual", 0)
        spl_txt = f"system_prompt_len={knobs['system_prompt_len']}" + (
            f" (actual {spl_actual})"
            if spl_actual != knobs["system_prompt_len"]
            else ""
        )
        ctx_txt = f"context[{knobs['min_context']},{knobs['max_context']}]"
        if knobs["max_context"] is not None:
            ctx_txt += (
                f" (model window: recorded-peak cap "
                f"{knobs['max_context'] - sys_len} after -{sys_len} sys)"
            )
        print(
            f"[plan] {len(plan)} sessions from {args.dataset_name} ({data_dir}), "
            f"{spl_txt}, "
            f"output_policy={knobs['output_policy']}, "
            f"turns[min={knobs['min_turns']},max={knobs['max_turns']}], "
            f"{ctx_txt}"
        )

    total_turns = sum(len(p.turns) for p in plan)
    total_out = sum(t.max_new_tokens for p in plan for t in p.turns)
    basis = "pinned per-turn lengths"
    total_gap_s = sum(t.pre_gap_ms for p in plan for t in p.turns) / 1000.0
    waves = len(plan) / max(1, args.max_concurrency)
    gap_per_slot_min = total_gap_s / max(1, args.max_concurrency) / 60.0
    print(
        f"[workload] {total_turns} turns, {total_out:,} output tokens "
        f"[{basis}], {total_gap_s / 60:.0f} min of tool waits, "
        f"{waves:.1f} waves over {args.max_concurrency} slots"
    )
    est = "  ".join(
        f"{tput}tok/s->{total_out / tput / 60 + gap_per_slot_min:.0f}min"
        for tput in (250, 1000, 4000)
    )
    print(f"[estimate] wall clock at assumed aggregate throughput: {est}")
    print("[estimate] the [progress] lines below carry a measured ETA.")

    sampling, force_lengths = _build_sampling(args, knobs)
    return plan, knobs, sampling, force_lengths


def _build_sampling(args: argparse.Namespace, knobs: dict) -> tuple:
    """Pinned sampling; returns (sampling dict, ignore_eos)."""
    sampling: dict = {"temperature": args.temperature}
    # Mirrors bench_serving: top_p is only sent when it actually
    # constrains sampling, avoiding divergent edge-case handling of an
    # explicit top_p=1.0 across backends.
    if args.top_p < 1.0:
        sampling["top_p"] = args.top_p
    if args.top_k is not None:
        sampling["top_k"] = args.top_k
    if args.presence_penalty is not None:
        sampling["presence_penalty"] = args.presence_penalty
    try:
        ctk = (
            json.loads(args.chat_template_kwargs)
            if args.chat_template_kwargs
            else None
        )
    except json.JSONDecodeError as e:
        raise SystemExit(f"bad --chat-template-kwargs: {e}")
    if ctk:
        sampling["chat_template_kwargs"] = ctk
    # Lengths are always pinned (trace records / fixed:K), so ignore_eos is
    # always on: without it a natural EOS would cut turns short.
    force_lengths = True
    extras = f" top_p={args.top_p}" if args.top_p < 1.0 else ""
    print(
        f"[sampling] temperature={args.temperature}{extras}"
        f" ignore_eos={force_lengths} chat_template_kwargs={ctk or '{}'}"
    )
    return sampling, force_lengths


def _execute_point(
    args: argparse.Namespace,
    plan,
    knobs: dict,
    tokenizer,
    backend: str = "sglang",
    max_model_len: Optional[int] = None,
) -> Any:
    """Flush, warm up, replay, and compute metrics for one run."""
    api_url = _resolve_api_url(args.base_url)
    sampling, force_lengths = _build_sampling(args, knobs)
    extra_headers = _parse_header_args(args.header)
    extra_request_body = _parse_extra_request_body(args.extra_request_body)

    async def _main() -> tuple:
        if args.flush_cache:
            await flush_cache(args.base_url, backend=backend)
        if args.warmup_requests > 0:
            await warmup(
                pool=plan,
                api_url=api_url,
                model=args.model,
                tokenizer=tokenizer,
                n=args.warmup_requests,
                disable_stream=args.disable_stream,
                extra_headers=extra_headers,
                extra_request_body=extra_request_body,
                backend=backend,
                max_duration_s=args.max_duration,
            )
        return await run_closed_loop(
            pool=plan,
            api_url=api_url,
            model=args.model,
            max_concurrency=args.max_concurrency,
            tokenizer=tokenizer,
            content_mode="trace",
            disable_stream=args.disable_stream,
            duration_s=args.duration,
            respect_gaps=not args.no_gaps,
            gap_scale=args.gap_scale,
            worker_header=args.worker_header,
            verbose=args.verbose,
            progress_interval=args.progress_interval,
            extra_body=sampling,
            ignore_eos=force_lengths,
            turn_timeout_s=args.turn_timeout,
            metrics_interval=args.metrics_interval,
            session_timeout_s=args.session_timeout,
            extra_headers=extra_headers,
            extra_request_body=extra_request_body,
            backend=backend,
            max_duration_s=args.max_duration,
            disable_tqdm=args.disable_tqdm,
            max_model_len=max_model_len,
        )

    results, server_snapshots = asyncio.run(_main())
    metrics = compute_metrics(
        sessions=results,
        target_concurrency=args.max_concurrency,
        steady_frac=args.steady_frac,
        steady_only=True,
    )
    summarize_server_metrics(metrics, server_snapshots, backend=backend)
    return metrics, results, server_snapshots


def cmd_run(args: argparse.Namespace) -> None:
    if args.dataset_name == "random" and args.dataset_path:
        print(
            "error: --dataset-name random synthesizes locally and reads no "
            "files; drop --dataset-path (or pick a real dataset name)",
            file=sys.stderr,
        )
        sys.exit(1)
    base_url = _resolve_base_url(args)
    args.base_url = base_url
    asyncio.run(wait_for_server_ready(base_url, args.ready_check_timeout_sec))
    backend = _detect_backend(base_url, args.backend)
    print(f"[backend] {backend}")
    if args.backend != "auto":
        _verify_declared_backend(base_url, backend)
    if args.flush_cache and backend == "openai":
        print(
            "[flush_cache] skipped: the openai backend has no "
            "cache-flush endpoint"
        )
        args.flush_cache = False
    max_model_len = _fetch_max_model_len(base_url)
    if max_model_len:
        print(
            f"[ctx-guard] model context {max_model_len} tokens; sessions "
            "whose estimated prompt + pinned completion exceed it stop early"
        )
    else:
        print("[ctx-guard] disabled: /v1/models reports no max_model_len")
    # /model_info is an SGLang endpoint; for other backends the tokenizer
    # resolves from --model (local dir) or --tokenizer instead.
    tokenizer = load_tokenizer(
        args.tokenizer or args.model,
        base_url if backend == "sglang" else None,
    )
    plan, knobs, _, _ = _prepare_plan(args, tokenizer)
    metrics, results, server_snapshots = _execute_point(
        args, plan, knobs, tokenizer, backend, max_model_len=max_model_len
    )
    if backend != "sglang":
        # host/device cache layering is an sglang-only concept
        # (not measurable != zero). Capacity and eviction are measured
        # for vllm too; openai has no /metrics so both stay None via the
        # empty-snapshot no-op in summarize_server_metrics.
        metrics.host_cache_ratio = None
        metrics.host_share_of_cache = None

    server_info = (
        asyncio.run(fetch_server_info(args.base_url))
        if backend == "sglang"
        else None
    )

    config = {
        "model": args.model,
        "backend": backend,
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "tag": args.tag,
        "system_prompt_len": knobs["system_prompt_len"],
        "system_prompt_len_actual": knobs.get("system_prompt_len_actual", 0),
        "max_model_len": max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "chat_template_kwargs": args.chat_template_kwargs,
        "ignore_eos": True,
        "min_context": knobs["min_context"],
        "max_context": knobs["max_context"],
        "output_details": args.output_details,
        "max_concurrency": args.max_concurrency,
        "num_sessions": metrics.num_sessions,
        "duration_s": args.duration,
        "respect_gaps": not args.no_gaps,
        "gap_scale": args.gap_scale,
        "seed": args.seed,
    }
    print_report(metrics, config)
    # bench_serving habit: results always land on disk. Without an explicit
    # --output-file, pick a dated, collision-resistant default name.
    output_file = args.output_file
    if not output_file:
        import datetime as _dt

        today = _dt.datetime.now().strftime("%m%d_%H%M%S")
        ds_label = args.dataset_name.replace("/", "-")
        output_file = f"{backend}_{today}_{ds_label}.jsonl"
    save_results(
        results,
        metrics,
        output_file,
        config,
        server_snapshots,
        server_info=server_info,
        include_details=args.output_details,
    )
    print(f"[save] appended to {output_file}")


# ----------------------------------------------------------------------
# data listing
# ----------------------------------------------------------------------
def _list_datasets() -> None:
    """Print the dataset x tier combinations `python3 -m abench data` fetches."""
    print("Available dataset x tier combinations (from ModelScope):")
    for spec in TIERS.values():
        size_mb = sum(f[1] for f in spec.files) / 1e6
        print(
            f"  {spec.name:<16s} {spec.sessions:>6d} sessions "
            f"({size_mb:.0f} MB)"
        )
    print()
    print("'random' needs no download: python3 -m abench run --dataset-name random")
