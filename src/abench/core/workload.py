# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Workload construction: turn the normalized DeepSWE sessions into a
concrete, parameterized benchmark plan.

This is where the four benchmark *knobs* live:

1. **System-prompt length** (``system_prompt_len``) — the original
   trace system prompt is stripped by ``dataset.py``; here we rebuild a
   shared system prompt tiled to an exact token length. Content is
   irrelevant to performance, so we simply repeat a canonical base block
   until we hit the target token count. The same bytes are used for
   every session, so the system prefix is maximally cacheable (matching
   production, where the system prompt is shared and static).

2. **Turns** (``min_turns`` / ``max_turns``) — a pure *filter/cap*
   knob. Sessions shorter than ``min_turns`` are dropped; sessions
   longer than ``max_turns`` are truncated *only* when ``max_turns`` is
   set. We never pad short sessions to a turn count.

3. **Peak context** (``min_context`` / ``max_context``) — an explicit
   token band on a session's peak recorded context. Stating both edges
   keeps it obvious which sessions a band admits, and lets the upper
   edge be pinned under the serving context window.

4. **Output length** (``output_policy``) — per-turn generated tokens.
   ``trace`` uses the recorded ``completion_tokens``; ``fixed:K`` emits
   K tokens/turn; ``budget:T`` spreads a total budget evenly across the
   session's turns (``T // num_turns``, floored at 1).

The output is a list of :class:`PlannedSession` objects consumed by the
closed-loop replay engine.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from abench.core.dataset import NormSession, NormTurn

# A neutral base block used to synthesize system prompts of arbitrary
# length. Content does not affect serving performance; only token count
# does. Repeating a fixed block keeps the prefix byte-identical (and thus
# fully cacheable) across every session.
_SYS_BASE = (
    "You are a helpful software engineering assistant operating inside a "
    "sandboxed repository. You can read files, run shell commands, and edit "
    "code to resolve the assigned task. Always inspect the relevant modules "
    "before making a change and verify your edits with the available tools. "
)

# Neutral text block for synthesizing per-turn user/tool messages.
_TURN_BASE = (
    "The user has provided additional context and tool output for this turn. "
    "Please review the following information and continue working on the task. "
)


# ----------------------------------------------------------------------
# Planned workload structures
# ----------------------------------------------------------------------
@dataclass
class PlannedTurn:
    """A turn with a resolved output length, ready to be issued."""

    index: int
    input_delta: List[Dict[str, str]]
    assistant_text: str
    max_new_tokens: int
    recorded_input_tokens: int
    recorded_cached_tokens: int
    pre_gap_ms: float
    # Recorded tool calls, carried through from NormTurn.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Kept so the report can check the achieved mean output length against
    # what the trace recorded, even when the policy does not force lengths.
    recorded_output_tokens: int = 0


@dataclass
class PlannedSession:
    """A session with a concrete system prompt and per-turn output plan."""

    session_id: str
    system_prompt: str
    turns: List[PlannedTurn]
    routing_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)


# ----------------------------------------------------------------------
# System-prompt tiling
# ----------------------------------------------------------------------
def measure_text_tokens(text: str, tokenizer: Optional[Any]) -> int:
    """Token count of ``text`` under ``tokenizer`` (chars/4 when absent)."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _trim_to_token_target(text: str, target_tokens: int, tokenizer) -> str:
    """Char-trim an over-long tiling until it encodes to ~``target_tokens``.

    decode(encode(t)) is not length-preserving under BPE — re-encoding a
    decoded string can merge differently — so a prompt built by decoding
    truncated ids lands short (an 8192-token target once landed at 5831). The
    robust construction is to keep an over-long tiling and trim characters
    while measuring. Callers must still report the measured length; the
    result is exact or within a couple of tokens.
    """
    cur = measure_text_tokens(text, tokenizer)
    guard = 0
    while cur > target_tokens and guard < 512:
        if cur - target_tokens <= 2:
            text = text[:-1]
        else:
            # Remove roughly the surplus in characters; 4 chars/token is
            # the pessimistic bound, so we approach from above.
            cut = max(1, (cur - target_tokens) * len(text) // (cur * 4))
            text = text[:-cut]
        cur = measure_text_tokens(text, tokenizer)
        guard += 1
    return text


def build_system_prompt(
    target_tokens: int,
    tokenizer: Optional[Any] = None,
    base: str = _SYS_BASE,
) -> str:
    """Return a system prompt tiled to ``target_tokens`` tokens.

    ``target_tokens == 0`` returns an empty string (no system prompt).
    ``target_tokens < 0`` is treated as "use the base block once".

    With a tokenizer the returned text encodes to the target length
    (within a couple of tokens); the caller should measure it with
    :func:`measure_text_tokens` and use that number for context
    accounting instead of assuming the target.
    """
    if target_tokens == 0:
        return ""
    if target_tokens < 0:
        return base

    base_tok = max(1, measure_text_tokens(base, tokenizer))
    # Tiling with margin: the trim pass only shrinks, so start above the
    # target (re-encoding a decoded tiling can compress by tens of
    # percent, so a bare 1x margin is not enough).
    reps = max(2, (target_tokens * 3) // base_tok + 4)
    text = base * reps
    if tokenizer is not None:
        guard = 0
        while measure_text_tokens(text, tokenizer) < target_tokens + base_tok:
            text = base * (reps * 2)
            reps *= 2
            guard += 1
            if guard > 8:
                break
        return _trim_to_token_target(text, target_tokens, tokenizer)

    # Char-based fallback: 4 chars/token heuristic.
    return text[: target_tokens * 4]


# ----------------------------------------------------------------------
# Output-length policy
# ----------------------------------------------------------------------
def make_output_policy(spec: str) -> Callable[[NormSession, NormTurn], int]:
    """Return a function mapping (session, turn) -> max_new_tokens.

    Every policy yields a length to *pin*: the engine sends it as
    ``max_completion_tokens`` together with ``ignore_eos``, so a turn
    never ends on the model's own EOS. There is deliberately no
    "let the model decide" policy — a reasoning model left to itself runs
    to the generation cap and emits many times the recorded work, which
    makes the run incomparable with the trace it came from.

    ``spec`` grammar:
      - ``trace``    : recorded completion_tokens (min 1)
      - ``fixed:K``  : constant K tokens/turn
      - ``budget:T`` : floor(T / num_turns), min 1, same for all turns
    """
    spec = (spec or "trace").strip()
    if spec == "trace":
        return lambda sess, turn: max(1, turn.output_tokens)
    if spec.startswith("fixed:"):
        k = max(1, int(spec.split(":", 1)[1]))
        return lambda sess, turn: k
    if spec.startswith("budget:"):
        total = max(1, int(spec.split(":", 1)[1]))
        return lambda sess, turn: max(1, total // max(1, sess.num_turns))
    raise ValueError(f"unknown output policy: {spec!r}")


# ----------------------------------------------------------------------
# Gap-distribution policy (synthetic mode)
# ----------------------------------------------------------------------
def make_gap_policy(spec: str, cap_ms: float = 30_000.0, seed: int = 42) -> Callable[[int], float]:
    """Return a function mapping ``turn_index`` -> ``gap_ms``.

    ``spec`` grammar:
      - ``zero``                : no gap (pure multi-turn chat)
      - ``fixed:Ms``            : constant M milliseconds
      - ``uniform:lo,hi``       : uniform random in [lo, hi]
      - ``exponential:mean``    : exponential with given mean (default)

    All values are capped at ``cap_ms``.  T1 (turn_index == 0) always
    returns 0 — the first turn has no preceding tool wait.
    """
    spec = (spec or "exponential:3000").strip()

    def _bad() -> ValueError:
        return ValueError(f"unknown gap policy: {spec!r}")

    if spec == "zero":
        return lambda idx: 0.0

    if spec.startswith("fixed:"):
        try:
            ms = float(spec.split(":", 1)[1])
        except ValueError:
            raise _bad() from None
        ms = min(ms, cap_ms)
        return lambda idx: 0.0 if idx == 0 else ms

    if spec.startswith("uniform:"):
        parts = spec.split(":", 1)[1].split(",")
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except ValueError:
            raise _bad() from None
        hi = min(hi, cap_ms)
        lo = min(lo, hi)
        rng = random.Random(seed)
        return lambda idx: 0.0 if idx == 0 else min(rng.uniform(lo, hi), cap_ms)

    if spec.startswith("exponential:"):
        try:
            mean = float(spec.split(":", 1)[1])
        except ValueError:
            raise _bad() from None
        rng = random.Random(seed)
        return lambda idx: 0.0 if idx == 0 else min(rng.expovariate(1.0 / mean), cap_ms)

    raise ValueError(f"unknown gap policy: {spec!r}")


# ----------------------------------------------------------------------
# Benchmark profiles (frozen reference configurations)
# ----------------------------------------------------------------------
@dataclass
class Profile:
    """A frozen, reproducible benchmark configuration.

    Profiles select real sessions by their **peak recorded context** and
    never truncate turns, so a tier keeps each trajectory's natural
    length. The output policy is an independent knob that always
    defaults to ``trace`` (the dataset's recorded output lengths).
    """

    name: str
    min_turns: int
    min_context: int
    max_context: int
    system_prompt_len: int
    description: str


# Bands are chosen from the full DeepSWE v1.1 index so every tier has a
# pool of thousands. ``large`` stops at 176K because 176K + an 8K system
# prompt still leaves ~18K of headroom under a 200K serving context
# window; sessions above that cannot be compared across models.
PROFILES: Dict[str, Profile] = {
    "mixed": Profile(
        name="mixed",
        min_turns=3,
        min_context=8_000,
        max_context=176_000,
        system_prompt_len=8_192,
        description=(
            "Whole usable band, 8-176K peak context (natural mix: 20% under "
            "48K, 56% 48-128K, 25% over 128K; ~48 turns). Primary scan — one "
            "pool, no rebalancing, as MLPerf does."
        ),
    ),
    "small": Profile(
        name="small",
        min_turns=3,
        min_context=8_000,
        max_context=48_000,
        system_prompt_len=8_192,
        description="Light prefill: 8-48K peak context, ~20 turns.",
    ),
    "mid": Profile(
        name="mid",
        min_turns=3,
        min_context=48_000,
        max_context=128_000,
        system_prompt_len=8_192,
        description="Moderate prefill: 48-128K peak context, ~48 turns.",
    ),
    "large": Profile(
        name="large",
        min_turns=3,
        min_context=128_000,
        max_context=176_000,
        system_prompt_len=8_192,
        description="Heavy prefill: 128-176K peak context, ~80 turns.",
    ),
}


# ----------------------------------------------------------------------
# Selection + planning
# ----------------------------------------------------------------------
def select_sessions(
    sessions: List[NormSession],
    min_turns: int = 1,
    max_turns: Optional[int] = None,
    min_context: Optional[int] = None,
    max_context: Optional[int] = None,
    num_sessions: Optional[int] = None,
    seed: int = 0,
    sys_prompt_tokens: int = 0,
) -> List[NormSession]:
    """Filter and sample sessions according to the turn/context knobs.

    * ``max_turns`` truncates long sessions (only when set).
    * ``min_turns`` drops sessions that are too short (never padded).
    * ``min_context`` / ``max_context`` keep sessions whose peak context
      falls inside the band. The band is tested *after* the optional
      ``max_turns`` cap, so it describes what will actually be replayed.
    * ``max_context`` is the MODEL context window (the user-facing
      contract): sessions are kept while their recorded peak fits the
      window left over after the injected prompt,
      ``recorded_peak <= max_context - sys_prompt_tokens``.
      ``min_context`` applies to the same replayed-peak axis
      (recorded peak + ``sys_prompt_tokens``). With no injection
      (``sys_prompt_tokens == 0``) both apply to the recorded peak alone.
    """
    if num_sessions is not None and num_sessions < 0:
        raise ValueError(f"num_sessions must be >= 0, got {num_sessions}")
    pool: List[NormSession] = []
    for s in sessions:
        if s.num_turns < min_turns:
            continue
        turns = s.turns
        if max_turns is not None and len(turns) > max_turns:
            turns = turns[:max_turns]
        cand = NormSession(
            session_id=s.session_id,
            system_prompt=s.system_prompt,
            turns=turns,
            source=s.source,
        )
        peak = cand.peak_input_tokens + sys_prompt_tokens
        if min_context is not None and peak < min_context:
            continue
        if max_context is not None and peak > max_context:
            continue
        pool.append(cand)

    rng = random.Random(seed)
    rng.shuffle(pool)
    if num_sessions is not None and num_sessions < len(pool):
        pool = pool[:num_sessions]
    return pool


_RATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "recording_rates.json"
)
# Last-resort value if the shipped table cannot be read at all; the table
# carries the authoritative pooled fit under "_pooled".
_POOLED_FALLBACK = 140.0
_rates_cache: Optional[Dict[str, float]] = None
_pooled_cache: Optional[float] = None


def _load_rates() -> None:
    global _rates_cache, _pooled_cache
    try:
        with open(_RATES_PATH) as fh:
            raw = json.load(fh)
        _rates_cache = {k: float(v) for k, v in raw.items() if not k.startswith("_")}
        _pooled_cache = float(raw.get("_pooled", _POOLED_FALLBACK))
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] cannot read {_RATES_PATH} ({exc}); every session falls back "
            f"to the pooled rate {_POOLED_FALLBACK} tok/s, so extracted tool "
            f"waits will not match runs made with the table",
            file=sys.stderr,
        )
        _rates_cache = {}
        _pooled_cache = _POOLED_FALLBACK


def _recording_rates() -> Dict[str, float]:
    """Per-source-model output rate (tokens/s) of the recorded runs."""
    if _rates_cache is None:
        _load_rates()
    return _rates_cache or {}


def _pooled_rate() -> float:
    if _pooled_cache is None:
        _load_rates()
    return _pooled_cache or _POOLED_FALLBACK


def _recording_rate(session: NormSession) -> float:
    """Estimate the output rate of the model that recorded ``session``.

    ``interval = tool_time + output_tokens / rate`` with tool time
    unobserved, so the rate is taken as the 5th percentile of
    ``interval / output_tokens`` — the turns with the least tool work.
    That percentile assumes those turns carry no tool work at all, which
    makes the rate an *upper* bound and the generation time removed below
    a *lower* bound: the split never eats into real tool idle.

    The percentile is read from a table fitted per source model over the
    whole corpus (tens of thousands of intervals each) rather than fitted
    per session. Within one session tool time correlates with output
    length, so a per-session regression is not identified: a Theil-Sen
    variant returned 32 tok/s at the 10th percentile of small-tier
    sessions and zeroed 26% of turns, against 88-295 tok/s per model.
    """
    model = str((session.source or {}).get("model", ""))
    rate = _recording_rates().get(model)
    if rate:
        return rate
    ratios = sorted(
        t.pre_gap_ms / 1000.0 / t.output_tokens
        for t in session.turns
        if t.output_tokens > 0 and t.pre_gap_ms > 0
    )
    if len(ratios) >= 20:
        return 1.0 / ratios[int(0.05 * len(ratios))]
    return _pooled_rate()


def tool_delays_ms(session: NormSession) -> List[float]:
    """Split each recorded turn-to-turn interval into its tool-wait part.

    ``NormTurn.pre_gap_ms`` is the delta between consecutive recorded agent
    steps, so it covers the tool execution *plus* one turn of the recording
    model's generation (the interval correlates with the *next* turn's
    output length at r=0.68, against r=0.16 for the previous turn's, so it
    is the generation of the turn the interval leads into). Charging that
    to the replayed server would pay for inference twice.

    Splitting it from a table rather than from the replay's own latency
    keeps the delay a property of the dataset alone, so the same
    trajectory offers identical idle time to every server under test — as
    MLPerf's dataset-side ``delay_seconds`` does. The split is a modelling
    assumption, not a measurement: the dataset has no per-tool duration
    field, and the only independent probe (turn 0, which has no tool call
    before it) is unusable because non-agent steps carry whole-second
    timestamps. Sanity bound: every fitted rate exceeds the index's
    ``n_output_tokens / agent_duration_seconds`` by 1.6-3.6x, as it must,
    since that ratio charges tool time to generation.
    """
    rate = _recording_rate(session)
    return [
        max(0.0, t.pre_gap_ms - 1000.0 * t.output_tokens / rate)
        for t in session.turns
    ]


def plan_workload(
    sessions: List[NormSession],
    system_prompt_len: int,
    output_policy: str,
    tokenizer: Optional[Any] = None,
    routing_key_fn: Optional[Callable[[NormSession], Optional[str]]] = None,
    system_prompt: Optional[str] = None,
) -> List[PlannedSession]:
    """Attach a concrete system prompt and per-turn output plan.

    The synthesized system prompt is shared (byte-identical) across all
    sessions so its KV is fully reusable — matching production. Pass
    ``system_prompt`` to reuse an already-built prompt (the CLI builds it
    once up front so its measured length can inform session selection).
    """
    if system_prompt is None:
        system_prompt = build_system_prompt(system_prompt_len, tokenizer)
    sys_len = measure_text_tokens(system_prompt, tokenizer)
    out_fn = make_output_policy(output_policy)

    planned: List[PlannedSession] = []
    for s in sessions:
        delays = tool_delays_ms(s)
        pturns = [
            PlannedTurn(
                index=t.index,
                input_delta=t.input_delta,
                assistant_text=t.assistant_text,
                max_new_tokens=out_fn(s, t),
                recorded_input_tokens=t.input_tokens,
                recorded_cached_tokens=t.cached_tokens,
                pre_gap_ms=delay,
                tool_calls=t.tool_calls,
                recorded_output_tokens=t.output_tokens,
            )
            for t, delay in zip(s.turns, delays)
        ]
        planned.append(
            PlannedSession(
                session_id=s.session_id,
                system_prompt=system_prompt,
                turns=pturns,
                routing_key=routing_key_fn(s) if routing_key_fn else None,
                metadata={
                    "num_turns": s.num_turns,
                    "peak_input_tokens": s.peak_input_tokens,
                    "system_prompt_tokens": sys_len,
                    "model": s.source.get("model", ""),
                },
            )
        )
    return planned


# ----------------------------------------------------------------------
# Synthetic workload generation (no dataset required)
# ----------------------------------------------------------------------
def _tile_text(target_tokens: int, base: str, tokenizer: Optional[Any]) -> str:
    """Tile ``base`` to approximately ``target_tokens`` tokens."""
    if target_tokens <= 0:
        return ""
    if tokenizer is not None:
        base_tok = max(1, measure_text_tokens(base, tokenizer))
        reps = max(2, (target_tokens * 3) // base_tok + 4)
        return _trim_to_token_target(base * reps, target_tokens, tokenizer)
    return base * max(1, target_tokens // max(1, len(base) // 4))


def synthesize_sessions(
    num_sessions: int,
    turns: int,
    input_increment: int,
    output_len: int,
    system_prompt_len: int,
    gap_policy: str = "exponential:3000",
    gap_cap_ms: float = 30_000.0,
    tokenizer: Optional[Any] = None,
    routing_key_fn: Optional[Callable[[str], Optional[str]]] = None,
    seed: int = 42,
) -> List[PlannedSession]:
    """Generate synthetic PlannedSessions without a dataset.

    Each session has ``turns`` turns.  Every turn adds ``input_increment``
    tokens of new context (simulating tool output / user follow-up), so peak
    context grows linearly.  Output is fixed at ``output_len`` tokens/turn.
    Inter-turn gaps follow ``gap_policy`` (T1 always has gap 0).
    """
    if num_sessions < 0:
        raise ValueError(f"num_sessions must be >= 0, got {num_sessions}")
    gap_fn = make_gap_policy(gap_policy, cap_ms=gap_cap_ms, seed=seed)
    system_prompt = build_system_prompt(system_prompt_len, tokenizer)
    user_block = _tile_text(input_increment, _TURN_BASE, tokenizer)
    assistant_block = _tile_text(output_len, _SYS_BASE, tokenizer)

    planned: List[PlannedSession] = []
    for si in range(num_sessions):
        sid = f"synthetic-{si:04d}"
        pturns: List[PlannedTurn] = []
        for ti in range(turns):
            pturns.append(
                PlannedTurn(
                    index=ti,
                    input_delta=[{"role": "user", "content": user_block}],
                    assistant_text=assistant_block,
                    max_new_tokens=output_len,
                    recorded_input_tokens=input_increment,
                    recorded_cached_tokens=0,
                    pre_gap_ms=gap_fn(ti),
                    recorded_output_tokens=output_len,
                )
            )
        rk = routing_key_fn(sid) if routing_key_fn else None
        planned.append(
            PlannedSession(
                session_id=sid,
                system_prompt=system_prompt,
                turns=pturns,
                routing_key=rk,
                metadata={
                    "synthetic": True,
                    "num_turns": turns,
                    "input_increment": input_increment,
                    "output_len": output_len,
                    "gap_policy": gap_policy,
                    "gap_cap_ms": gap_cap_ms,
                },
            )
        )
    return planned
