# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Domain layer: session data, workload planning, result records.

The data model and the pure functions over it: no inference servers, no
telemetry, no command line. One exception — ``dataset`` fetches tier
files from the distribution mirror over plain ``urllib``, so that a bare
install can still read downloaded data.
"""

from abench.core.dataset import (
    TIERS,
    NormSession,
    NormTurn,
    TierSpec,
    download_tier,
    iter_jsonl_sessions,
    load_sessions,
    resolve_url,
    write_tier_manifest,
)
from abench.core.metrics import (
    AgenticMetrics,
    BucketStats,
    SessionRecord,
    TurnRecord,
    compute_metrics,
    summarize_server_metrics,
)
from abench.core.workload import (
    PROFILES,
    PlannedSession,
    PlannedTurn,
    Profile,
    build_system_prompt,
    make_gap_policy,
    make_output_policy,
    measure_text_tokens,
    plan_workload,
    select_sessions,
    synthesize_sessions,
)

__all__ = [
    "TIERS",
    "TierSpec",
    "NormSession",
    "NormTurn",
    "download_tier",
    "iter_jsonl_sessions",
    "load_sessions",
    "resolve_url",
    "write_tier_manifest",
    "AgenticMetrics",
    "BucketStats",
    "SessionRecord",
    "TurnRecord",
    "compute_metrics",
    "summarize_server_metrics",
    "PROFILES",
    "PlannedSession",
    "PlannedTurn",
    "Profile",
    "build_system_prompt",
    "make_gap_policy",
    "make_output_policy",
    "measure_text_tokens",
    "plan_workload",
    "select_sessions",
    "synthesize_sessions",
]
