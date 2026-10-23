# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Closed-loop replay engine.

Owns the concurrency model (fixed active sessions, per-turn sequencing),
tool-wait gaps, pinned decode lengths, warmup and the run fuse. Turn-level
HTTP plumbing is in ``abench.replay.transport``; server discovery and
telemetry are in ``abench.serve``.
"""

from abench.replay.engine import run_closed_loop, warmup

__all__ = ["run_closed_loop", "warmup"]
