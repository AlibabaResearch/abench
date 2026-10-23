# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, Alibaba Group and its affiliates

"""Engine-facing layer: everything that talks to the server under test.

``endpoints``  discovery, auth, readiness, tokenizer resolution, cache flush
``telemetry``  /metrics scraping and Prometheus parsing
``report``     terminal report and JSONL persistence
"""

from abench.serve.endpoints import (
    fetch_server_info,
    flush_cache,
    get_auth_headers,
    load_tokenizer,
    server_root,
    wait_for_server_ready,
)
from abench.serve.report import metrics_to_dict, print_report, save_results

__all__ = [
    "fetch_server_info",
    "flush_cache",
    "get_auth_headers",
    "load_tokenizer",
    "server_root",
    "wait_for_server_ready",
    "metrics_to_dict",
    "print_report",
    "save_results",
]
