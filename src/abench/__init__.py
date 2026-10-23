# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Abench — agentic inference benchmark.

Closed-loop replay of recorded agent trajectories (DeepSWE v1.1, and an
LMCache-derived Otel tier) against any OpenAI-compatible inference
server, plus a synthetic random workload for engines and context bands
the traces do not cover. System-prompt length, session selection and
the steady-state measurement window are all parameterized.

Decode lengths are pinned rather than left to the model: a replayed turn
generates its recorded ``completion_tokens``, a random turn generates
``--output-len``. Pinning is what makes two engines comparable on the
same workload; the report checks the achieved mean output length against
the recorded one so a server that ignores the pin is visible.

Datasets are distributed via the ABench ModelScope mirror
(Apsara-Stack/ABench).

Layering: ``cli`` (arguments) -> ``runner`` (orchestration) ->
``replay`` (closed-loop engine) -> ``serve`` (server discovery,
telemetry, reporting) -> ``core`` (data model, planning, metrics).
"""

from abench.core.dataset import (
    TIERS,
    NormSession,
    NormTurn,
    download_tier,
    iter_jsonl_sessions,
    load_sessions,
    resolve_url,
    write_tier_manifest,
)
from ._version import __version__
from abench.core.metrics import (
    AgenticMetrics,
    BucketStats,
    SessionRecord,
    TurnRecord,
    compute_metrics,
    summarize_server_metrics,
)
from abench.core.metrics import ServerMetricsSnapshot
from abench.replay.engine import run_closed_loop, warmup
from abench.serve.endpoints import (
    fetch_server_info,
    flush_cache,
    get_auth_headers,
    server_root,
    wait_for_server_ready,
)
from abench.serve.report import metrics_to_dict, print_report, save_results
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
    "__version__",
    "TIERS",
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
    "ServerMetricsSnapshot",
    "fetch_server_info",
    "flush_cache",
    "get_auth_headers",
    "run_closed_loop",
    "server_root",
    "wait_for_server_ready",
    "warmup",
    "metrics_to_dict",
    "print_report",
    "save_results",
    "PROFILES",
    "PlannedSession",
    "PlannedTurn",
    "Profile",
    "build_system_prompt",
    "measure_text_tokens",
    "make_output_policy",
    "make_gap_policy",
    "plan_workload",
    "select_sessions",
    "synthesize_sessions",
]
