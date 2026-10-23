# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Engine-facing HTTP surface: discovery, auth, readiness, cache control.

Everything that talks to the server *around* the measured turns lives here:
backend probing and port resolution, the tokenizer resolution chain,
auth headers, readiness polling, ``/model_info`` / ``/v1/models`` lookups,
request-shaping helpers, and cache flush. The replay engine and the runner
both build on this module; it never imports from them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import aiohttp

# ----------------------------------------------------------------------
# Auth + URL helpers
# ----------------------------------------------------------------------
def get_auth_headers() -> Dict[str, str]:
    """Auth headers from the environment, mirroring bench_serving.

    ``OPENAI_API_KEY`` wins and is sent as a Bearer token; ``API_KEY`` is
    sent verbatim (custom gateway schemes). Neither set -> no auth header.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return {"Authorization": f"Bearer {key}"}
    key = os.environ.get("API_KEY")
    if key:
        return {"Authorization": key}
    return {}


def server_root(url: str) -> str:
    """Strip any OpenAI route suffix, leaving the server's root URL."""
    base = url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


async def wait_for_server_ready(base_url: str, timeout_s: float) -> None:
    """Poll ``/v1/models`` until it answers, or fail with a clear error.

    ``timeout_s <= 0`` skips the check. Exits non-zero on timeout so a
    still-loading server is reported up front instead of surfacing as a
    wall of per-request timeouts.
    """
    if timeout_s <= 0:
        return
    url = server_root(base_url) + "/v1/models"
    headers = get_auth_headers()
    deadline = time.perf_counter() + timeout_s
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as http:
        while True:
            try:
                async with http.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            if time.perf_counter() >= deadline:
                print(
                    f"[ready] server not answering at {url} after "
                    f"{timeout_s:.0f}s; is it up? (raise "
                    f"--ready-check-timeout-sec, or 0 to skip)",
                    file=sys.stderr,
                )
                sys.exit(1)
            await asyncio.sleep(2.0)


async def fetch_server_info(base_url: str) -> Optional[dict]:
    """GET ``/server_info`` (SGLang-only); ``None`` on any failure."""
    url = server_root(base_url) + "/server_info"
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as http:
            async with http.get(url, headers=get_auth_headers()) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
    except Exception:
        pass
    return None


async def flush_cache(
    base_url: str, backend: str = "sglang"
) -> None:
    # Both endpoints live at the server root, not under the OpenAI prefix.
    # sglang: POST /flush_cache. vLLM: POST /reset_prefix_cache, gated
    # behind VLLM_SERVER_DEV_MODE=1 on the server (>= v0.10); a 404 means
    # the flag is off, which is reported as a hint rather than an error.
    if backend == "vllm":
        path = "/reset_prefix_cache"
    else:
        path = "/flush_cache"
    url = server_root(base_url) + path
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as s:
            async with s.post(url, headers=get_auth_headers()) as r:
                await r.read()
                if r.status == 404 and backend == "vllm":
                    print(
                        "[flush_cache] /reset_prefix_cache unavailable (404): "
                        "vLLM gates it behind VLLM_SERVER_DEV_MODE=1; "
                        "continuing without a cache flush"
                    )
                elif r.status != 200:
                    print(f"[flush_cache] warning: status={r.status} url={url}")
                else:
                    # Always announce success: a silent flush is
                    # indistinguishable from a no-op.
                    print(f"[flush_cache] status={r.status} ({path})")
    except Exception as exc:  # noqa: BLE001
        print(f"[flush_cache] warning: failed ({exc}); cache may be warm")


# ----------------------------------------------------------------------
# Backend detection (observation-layer adaptation)
# ----------------------------------------------------------------------
def _detect_backend(base_url: str, requested: str) -> str:
    """Resolve the serving backend: explicit, or probed from the server.

    ``auto`` probes: a ``/metrics`` body containing ``vllm:`` means vLLM;
    otherwise ``/server_info`` answering 200 means SGLang; anything else
    falls back to the generic OpenAI profile. vLLM is checked first
    because its dev-mode servers (VLLM_SERVER_DEV_MODE=1, required for
    /reset_prefix_cache) also expose a 200 /server_info — the probe
    signal must be the metrics namespace, not endpoint presence. The
    backend gates SGLang-only observation (cache-token breakdown,
    /metrics gauge names, /flush_cache, /model_info) — request
    compatibility alone does not make the results comparable.
    """
    if requested != "auto":
        return requested
    import urllib.request

    root = server_root(base_url)
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=5) as resp:
            if resp.status == 200 and b"vllm:" in resp.read():
                return "vllm"
    except Exception:
        pass
    try:
        with urllib.request.urlopen(root + "/server_info", timeout=5) as resp:
            if resp.status == 200:
                return "sglang"
    except Exception:
        pass
    return "openai"


# Conventional serving ports, mirroring bench_serving's lookup table.
_BACKEND_DEFAULT_PORTS = {"sglang": 30000, "openai": 30000, "vllm": 8000}


def _resolve_base_url(args: argparse.Namespace) -> str:
    """Compute the server base URL: --base-url wins, else --host/--port.

    Port resolution order (bench_serving semantics):

    1. explicit ``--port``;
    2. the declared backend's conventional port
       (sglang/openai 30000, vllm 8000);
    3. ``auto`` backend: try 30000 first; if the probe identifies vllm,
       re-probe on 8000. Both ports dead -> error naming both candidates.
    """
    if args.base_url:
        return args.base_url

    host = args.host or "0.0.0.0"
    if args.port is not None:
        return f"http://{host}:{args.port}"

    if args.backend != "auto":
        port = _BACKEND_DEFAULT_PORTS.get(args.backend, 30000)
        print(f"[url] --port unset, backend={args.backend} -> :{port}")
        return f"http://{host}:{port}"

    # auto: probe the sglang port, fall back to the vllm port.
    tried = []
    for port in (30000, 8000):
        url = f"http://{host}:{port}"
        tried.append(port)
        backend = _detect_backend(url, "auto")
        if backend != "openai":
            print(
                f"[url] auto-detected backend={backend} on {url} "
                f"(tried {', '.join(str(p) for p in tried)})"
            )
            return url
    print(
        f"error: no inference server found on {host} (tried ports "
        f"{', '.join(str(p) for p in tried)}); pass --port/--base-url "
        "for a non-default deployment",
        file=sys.stderr,
    )
    print(
        "if the server is still loading, re-run in a moment (readiness is "
        "not probed before auto-detection); for non-standard ports pass --port",
        file=sys.stderr,
    )
    sys.exit(1)


def _verify_declared_backend(base_url: str, backend: str) -> None:
    """Warn when an explicitly declared backend does not match the server.

    A wrong declaration is not fatal — the observation layer just runs
    with that backend's gates and the matching metrics silently go
    missing — so the warning is the only signal the user gets.
    """
    import urllib.request

    root = server_root(base_url)
    mismatch = False
    if backend == "sglang":
        try:
            with urllib.request.urlopen(root + "/server_info", timeout=5) as r:
                mismatch = r.status != 200
        except Exception:
            mismatch = True
    elif backend == "vllm":
        try:
            with urllib.request.urlopen(root + "/metrics", timeout=5) as r:
                mismatch = not (r.status == 200 and b"vllm:" in r.read())
        except Exception:
            mismatch = True
    # openai is the unverified fallback profile: nothing to check.
    if mismatch:
        probe = "/server_info" if backend == "sglang" else "/metrics"
        print(
            f"[warn] declared backend={backend} but {probe} probe does not "
            f"match; cache/KV observation metrics may be silently missing. "
            "Consider omitting --backend to let abench auto-detect.",
            file=sys.stderr,
        )


def _fetch_max_model_len(base_url: str) -> Optional[int]:
    """Best-effort read of the model context window from GET /v1/models.

    Both sglang and vLLM report ``max_model_len`` there; generic OpenAI
    endpoints usually do not (guard disabled -> None).
    """
    import urllib.request

    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/models", headers=get_auth_headers()
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("data", []):
            v = m.get("max_model_len")
            if isinstance(v, int) and v > 0:
                return v
    except Exception:
        pass
    return None


def _parse_header_args(pairs: Optional[List[str]]) -> dict:
    """Parse repeated ``--header Key=value`` entries into a dict."""
    headers: dict = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"bad --header {p!r}; use Key=value")
        k, _, v = p.partition("=")
        if not k.strip():
            raise SystemExit(f"bad --header {p!r}; header name must not be empty")
        headers[k.strip()] = v
    return headers


def _parse_extra_request_body(spec: Optional[str]) -> Optional[dict]:
    """Parse ``--extra-request-body`` JSON; must be a JSON object."""
    if not spec:
        return None
    try:
        obj = json.loads(spec)
    except json.JSONDecodeError as e:
        raise SystemExit(f"bad --extra-request-body: {e}")
    if not isinstance(obj, dict):
        raise SystemExit("--extra-request-body must be a JSON object")
    return obj


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------
def _model_info_tokenizer(base_url: Optional[str]) -> Optional[str]:
    """Ask the running server for its tokenizer path via GET /model_info.

    Mirrors sglang's bench_serving: the server already knows where its own
    tokenizer lives, so the user never has to pass --tokenizer by hand.
    Only useful when bench and server share a filesystem (the path must
    also exist locally). Best-effort with a short timeout.
    """
    if not base_url:
        return None
    import urllib.request

    url = base_url.rstrip("/") + "/model_info"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                return None
            info = json.loads(resp.read().decode())
        tok = info.get("tokenizer_path") or info.get("model_path")
        if tok and os.path.isdir(tok):
            return tok
        return None
    except Exception:
        return None


def _looks_like_local_model_dir(name: Optional[str]) -> bool:
    """True when `name` is a filesystem directory with tokenizer files."""
    if not name or name == "default":
        return False
    if not os.path.isdir(name):
        return False
    return any(
        os.path.exists(os.path.join(name, f))
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
    )


def load_tokenizer(
    name: Optional[str], base_url: Optional[str] = None
) -> Optional[Any]:
    """Resolve a tokenizer with the fallback chain:

    1. --tokenizer / --model that is a local dir with tokenizer files
    2. the running server's /model_info endpoint (same-machine servers)
    3. a non-local --model / --tokenizer name handed to from_pretrained
       (an HF repo id; fetched from the Hub, as bench_serving does)
    4. tiktoken cl100k_base as a rough generic fallback, then chars/4

    Matches bench_serving's resolution; the tiktoken/chars tail keeps an
    unreachable-Hub machine from hard-failing.
    """
    tok_id = None
    if name and _looks_like_local_model_dir(name):
        tok_id = name
    if tok_id is None:
        tok_id = _model_info_tokenizer(base_url)
    if tok_id is None and name and name != "default":
        # bench_serving parity: a non-local name (an HF repo id) is handed
        # to from_pretrained, which fetches it from the Hub.
        tok_id = name
    if tok_id is not None:
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
        except Exception:
            pass
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _resolve_api_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"
