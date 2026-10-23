# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Dataset acquisition and loading for the ABench ModelScope mirror.

The ABench datasets live on ModelScope
(https://www.modelscope.cn/datasets/Apsara-Stack/ABench). Each tier is
one or more Zstandard-compressed JSONL files; every line is a complete
normalized session in the ``abench-session-jsonl-v1`` schema::

    {
      "session_id": str,
      "source": {"dataset", "agent", "model", "trial"},
      "system_prompt": str,
      "turns": [
        {
          "index": int,
          "input_delta": [{"role", "content"}],  # msgs added before this turn
          "assistant_text": str,     # recorded assistant output (for trace mode)
          "input_tokens": int,       # authoritative prompt_tokens
          "output_tokens": int,      # authoritative completion_tokens
          "cached_tokens": int,      # authoritative cached prefix tokens
          "pre_gap_ms": float        # tool-wait before this turn
        }, ...
      ]
    }

This module downloads those files (resume + SHA-256 verified) and loads
them back as :class:`NormSession` objects. The dataset x tier catalog is
hardcoded below per the packaging policy: update it when a new version
is published.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

MODELSCOPE_REPO = "Apsara-Stack/ABench"
RESOLVE_TEMPLATE = (
    "https://www.modelscope.cn/datasets/{repo}/resolve/master/{path}"
)

# Cap tool-wait gaps so a single stalled tool call cannot dominate the
# closed-loop schedule. The real distribution has a long tail; 30 s keeps
# replays tractable while preserving the "agent is idle" semantics.
DEFAULT_GAP_CAP_MS = 30_000.0


@dataclass(frozen=True)
class TierSpec:
    """One downloadable dataset x tier combination."""

    dataset: str
    tier: str
    version: str
    sessions: int
    band: Optional[List[int]] = None
    system_prompt_len: Optional[int] = None
    files: tuple = ()  # ((repo_path, size_bytes, sha256), ...)

    @property
    def name(self) -> str:
        return f"{self.dataset}/{self.tier}"


# The packaged catalog, mirroring catalog.json on ModelScope. Tiers not
# listed here are not published; update this table when the mirror is.
TIERS: Dict[str, TierSpec] = {
    t.name: t
    for t in (
        TierSpec(
            dataset="deepswe",
            tier="mixed",
            version="v1.1",
            sessions=800,
            band=[8_000, 176_000],
            system_prompt_len=8_192,
            files=(
                (
                    "data/deepswe/v1.1/mixed/data.jsonl.zst",
                    41_916_286,
                    "242179e664f883ab7a9acc5775f7cd81d1a5c3dda07ba7495257cfe44c2c652f",
                ),
            ),
        ),
        TierSpec(
            dataset="deepswe",
            tier="full",
            version="v1.1",
            sessions=22_489,
            system_prompt_len=8_192,
            files=(
                (
                    "data/deepswe/v1.1/full/part-00000-of-00004.jsonl.zst",
                    453_056_964,
                    "158d7455419c46d072e8e23eb6308999d24c158030bcd920d2fc567ed40d4c92",
                ),
                (
                    "data/deepswe/v1.1/full/part-00001-of-00004.jsonl.zst",
                    446_134_353,
                    "e07772cb2af49c0687a10c253a569ec2c608008cd022dfcdbe88a0dd8c733a55",
                ),
                (
                    "data/deepswe/v1.1/full/part-00002-of-00004.jsonl.zst",
                    446_203_059,
                    "c40251439a7c43e90038bb5427a82f2cd5c053ba25bdaa69870f73ee8303f775",
                ),
                (
                    "data/deepswe/v1.1/full/part-00003-of-00004.jsonl.zst",
                    479_975_752,
                    "342c4e7aa4bac5b1e350dd3eb5e533f0c30bc675e8b88158ee7898dccb387af1",
                ),
            ),
        ),
        TierSpec(
            dataset="lmcache",
            tier="full",
            version="v1",
            sessions=767,
            files=(
                (
                    "data/lmcache/v1/full/data.jsonl.zst",
                    9_059_217,
                    "fd190601d4a3b690d24d5c416eb87a5e52c341f8b5658a76371e605f8e1a2ff5",
                ),
            ),
        ),
    )
}


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class NormTurn:
    """One recorded LLM request/response inside a trajectory."""

    index: int
    input_delta: List[Dict[str, str]]
    assistant_text: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    pre_gap_ms: float = 0.0
    # Recorded tool calls, kept structured in addition to being folded into
    # ``assistant_text`` (some downstream analyses want them parsed).
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "input_delta": self.input_delta,
            "assistant_text": self.assistant_text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "pre_gap_ms": self.pre_gap_ms,
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormTurn":
        return cls(
            index=d["index"],
            input_delta=d["input_delta"],
            assistant_text=d.get("assistant_text", ""),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            cached_tokens=d.get("cached_tokens", 0),
            pre_gap_ms=d.get("pre_gap_ms", 0.0),
            tool_calls=d.get("tool_calls") or [],
        )


@dataclass
class NormSession:
    """A normalized agent trajectory ready for replay."""

    session_id: str
    system_prompt: str
    turns: List[NormTurn]
    source: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def peak_input_tokens(self) -> int:
        return max((t.input_tokens for t in self.turns), default=0)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "system_prompt": self.system_prompt,
            "turns": [t.to_dict() for t in self.turns],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormSession":
        return cls(
            session_id=d["session_id"],
            system_prompt=d.get("system_prompt", ""),
            turns=[NormTurn.from_dict(t) for t in d.get("turns", [])],
            source=d.get("source", {}),
        )


# ----------------------------------------------------------------------
# Download (ModelScope)
# ----------------------------------------------------------------------
def resolve_url(repo_path: str, repo: str = MODELSCOPE_REPO) -> str:
    """Direct-download URL for one file of the ABench ModelScope mirror."""
    return RESOLVE_TEMPLATE.format(repo=repo, path=repo_path)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_tier(
    spec: TierSpec,
    out_dir: str,
    repo: str = MODELSCOPE_REPO,
    verbose: bool = True,
) -> List[str]:
    """Download every file of ``spec`` into ``out_dir`` (resume + verify).

    Each file lands under its relative name (subdirectories included).
    Existing files with the expected size and SHA-256 are kept, so an
    interrupted download resumes instead of restarting. Returns the list
    of local file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    done: List[str] = []
    for repo_path, size, sha in spec.files:
        dst = os.path.join(out_dir, os.path.basename(repo_path))
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            if _sha256_file(dst) == sha:
                done.append(dst)
                if verbose:
                    print(f"[data] cached: {os.path.basename(dst)}")
                continue
            os.remove(dst)  # corrupt partial -> redownload
        url = resolve_url(repo_path, repo=repo)
        if verbose:
            print(
                f"[data] downloading {os.path.basename(dst)} "
                f"({size / 1e6:.1f} MB)"
            )
        tmp = dst + ".part"
        _download_file(url, tmp, expected_size=size)
        got = _sha256_file(tmp)
        if got != sha:
            os.remove(tmp)
            raise RuntimeError(
                f"sha256 mismatch for {repo_path}: expected {sha}, got {got}"
            )
        os.replace(tmp, dst)
        done.append(dst)
    return done


def _download_file(url: str, dst: str, expected_size: int) -> None:
    """Stream ``url`` to ``dst`` with a progress line, resuming if partial."""
    resume_from = 0
    if os.path.exists(dst):
        resume_from = os.path.getsize(dst)
        if resume_from >= expected_size:
            os.remove(dst)
            resume_from = 0
    req = urllib.request.Request(url)
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")
    mode = "ab" if resume_from else "wb"
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, mode) as f:
        total = int(r.headers.get("Content-Length", 0)) + resume_from
        got = resume_from
        last_pct = -10
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = int(got * 100 / total)
                if pct >= last_pct + 10:
                    last_pct = pct
                    print(f"  [{pct:3d}%] {got / 1e6:.1f}/{total / 1e6:.1f} MB")
    if got != expected_size:
        raise RuntimeError(
            f"download incomplete: {got} bytes, expected {expected_size}"
        )


def write_tier_manifest(
    spec: TierSpec,
    out_dir: str,
    files: Iterable[str],
) -> None:
    """Write the ``manifest.json`` the run side reads for band knobs."""
    manifest = {
        "dataset": spec.dataset,
        "tier": spec.tier,
        "version": spec.version,
        "n_sessions": spec.sessions,
        "band": spec.band,
        "system_prompt_len": spec.system_prompt_len,
        "files": [
            {"path": p, "size_bytes": os.path.getsize(p), "sha256": _sha256_file(p)}
            for p in files
        ],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def _open_jsonl_zst(path: str):
    """Open a (possibly zstd-compressed) JSONL file as a line iterator.

    Prefers the ``zstandard`` package; falls back to the ``zstd`` CLI so
    a bare install (aiohttp only) can still read downloaded data.
    """
    if not path.endswith(".zst"):
        return open(path, "r", encoding="utf-8")
    try:
        import zstandard as zstd

        return zstd.open(path, "rt", encoding="utf-8")
    except ImportError:
        import subprocess

        proc = subprocess.Popen(
            ["zstd", "-dc", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # Wrap the byte stream as text; process cleanup happens at GC/exhaust.
        return _TextProcessReader(proc)


class _TextProcessReader:
    """Minimal text-mode adapter over a subprocess stdout pipe."""

    def __init__(self, proc):
        import io

        self._proc = proc
        self._io = io.TextIOWrapper(proc.stdout, encoding="utf-8")

    def __iter__(self):
        return self._io.__iter__()

    def close(self):
        try:
            self._io.close()
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def iter_jsonl_sessions(paths: Iterable[str]):
    """Yield one raw session dict per JSONL line across ``paths``."""
    for path in paths:
        with _open_jsonl_zst(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_sessions(data_dir: str) -> List[NormSession]:
    """Load every session from a downloaded tier directory.

    Accepts both the v0.2 layout (``*.jsonl.zst`` JSONL files written by
    ``python3 -m abench data``) and a directory of per-session ``*.json`` files.
    """
    sessions: List[NormSession] = []
    if not os.path.isdir(data_dir):
        return sessions
    names = sorted(os.listdir(data_dir))
    jsonl = [
        os.path.join(data_dir, n)
        for n in names
        if n.endswith(".jsonl") or n.endswith(".jsonl.zst")
    ]
    if jsonl:
        for raw in iter_jsonl_sessions(jsonl):
            sessions.append(NormSession.from_dict(raw))
        return sessions
    for fn in names:
        if not fn.endswith(".json") or fn == "manifest.json":
            continue
        with open(os.path.join(data_dir, fn)) as f:
            sessions.append(NormSession.from_dict(json.load(f)))
    return sessions
