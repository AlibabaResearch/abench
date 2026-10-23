# Abench

[English](README.md) | 简体中文

> 本文是 [README.md](README.md) 的中文翻译;如有出入,以英文版为准。

Agentic 推理基准:将录制好的 agent 轨迹(**DeepSWE v1.1**、**LMCache**),或本地生成的负载,以闭环方式回放到任意 OpenAI 兼容推理服务(SGLang、vLLM 等)。

它回答:**单个 agent 推进得有多快?**——在固定并发、固定且可复现的 agent 负载下,给出逐轮 TTFT / TPOT / E2E、缓存行为与吞吐。

## 安装

```bash
pip install agentic-bench          # 从 PyPI 安装(首次发布后可用)
```

或安装 GitHub Release 中附带的 wheel:

```bash
pip install https://github.com/AlibabaResearch/abench/releases/download/v0.2.0/agentic_bench-0.2.0-py3-none-any.whl
```

从源码安装:

```bash
pip install .
```

PyPI 分发名为 `agentic-bench`;Python 模块和 `abench` 命令仍保持项目名。PyPI 上还有一个无关的 `abench` 包(ActivitySim),它也会安装 `abench` 模块和命令,因此不要在同一环境中同时安装两者。如果已经装了,先执行 `pip uninstall abench`。

## 快速开始

两个子命令。

```bash
# 1) 一次性数据准备:从 ABench ModelScope 镜像下载 deepswe/mixed
#    (~42 MB,带 SHA-256 校验、支持断点续传);数据落在
#    ~/abench_data/deepswe/mixed/
python3 -m abench data deepswe/mixed

# 2) 在固定并发下回放;每个 session 都会跑完(或受 --duration 限制),
#    并输出完整指标。--output-file 会为每次运行追加一行 JSON;
#    默认包含逐轮记录(--no-output-details 可精简输出)。
python3 -m abench run \
    --model default \
    --dataset-name deepswe/mixed --max-concurrency 16 \
    --output-file results.jsonl
# 通常不需要 --host/--port:会自动探测 localhost :30000(sglang)
# 和 :8000(vllm);只有非默认部署才需要 --port/--base-url
```

无需下载的冒烟测试(本地生成随机负载):

```bash
python3 -m abench run \
    --dataset-name random \
    --turns 3 --output-len 64 --max-concurrency 4
```

`python3 -m abench ...` 是主要调用方式(与 `bench_serving` 的调用方式一致)。wheel 也会安装等价的 `abench` 命令。结果始终落盘:不指定 `--output-file` 时,会在当前目录写入 `{backend}_{date}_{dataset}.jsonl`。

## 后端与认证

任何 OpenAI 兼容服务都可以;`--backend auto|sglang|vllm|openai`(默认 `auto`,从服务端探测)会适配**观测层**——仅 SGLang 有的扩展(逐层缓存明细、`/flush_cache`、`/metrics` 指标名)按后端启用,不支持的指标输出 `null` 而不是 `0`。请求格式始终是标准 OpenAI 格式。

**不要显式设置 `--backend` 和 `--port`**:`auto` 会探测引擎并解析端口(sglang/openai 30000,vLLM 8000)。显式指定错误的 `--backend` 不会报错——它会静默丢弃该后端的指标(缓存明细 / KV 池 / server_info);当声明的 backend 与服务端不匹配时,`abench` 会发出警告。

- **认证**:`export OPENAI_API_KEY=...`(自定义网关 scheme 用 `API_KEY=`)——会自动附加到每个请求,不需要 CLI 参数。
- **扩展入口**:`--header Key=Value`(可重复)用于自定义 HTTP 头;`--extra-request-body '{"seed": 123}'` 会合并进每个请求体,并可覆盖内置字段。
- `--ready-check-timeout-sec` 会在运行前轮询 `/v1/models`,让仍在加载的服务快速失败,而不是让每个请求都超时。

## CLI

| 命令 | 用途 |
|---|---|
| `python3 -m abench data <DATASET>/<TIER>` | 唯一的数据准备入口:从 [ABench ModelScope 镜像](https://www.modelscope.cn/datasets/Apsara-Stack/ABench)下载预设数据集 tier(带 SHA-256 校验、支持断点续传)到 `<data-path>/<dataset>/<tier>/`(`--list` 列出全部;只写数据集名而不写 tier 会被拒绝) |
| `python3 -m abench run` | 唯一的压测入口:固定并发下的闭环回放;数据由 `--dataset-name`(身份,在 `~/abench_data` 下查找)和/或 `--dataset-path`(文件 / tier 目录 / 根目录)定位 |

`--help` 显示核心参数;`--help-advanced` 显示全部。tokenizer 会自动解析(`--tokenizer`,否则本地 `--model` 目录,否则服务端 `/model_info`,否则对非本地名称使用 `AutoTokenizer.from_pretrained`,否则 `tiktoken`,最后 chars/4)——Hub 拉取行为与 `bench_serving` 一致;离线机器请设置 `HF_HUB_OFFLINE=1`。

## 文档

- [docs/README.md](docs/README.md) — 参考:闭环协议、数据集 tier、全部参数、指标定义(英文)
- [docs/DESIGN.md](docs/DESIGN.md) — 设计取舍与测量决策(英文)
- [CONTRIBUTING.md](CONTRIBUTING.md) — 如何贡献、CLA 与评审预期
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## 开发

```
src/abench/
  cli.py          参数解析与分发(python3 -m abench)
  runner.py       `data` 和 `run` 的实际逻辑
  core/           领域层:session 数据、负载规划、结果记录
  serve/          被测服务:端点、telemetry、报告
  replay/         闭环引擎与轮次传输
docs/             参考文档与设计说明
```

```bash
pip install -e .
python -m build --wheel   # 只打包 Python 包
```

## 许可证

Copyright (c) 2026, Alibaba Group and its affiliates. Abench 采用 Apache License 2.0——见 [LICENSE](LICENSE)。本项目 CLI 遵循的上游约定,以及数据集来源,记录在 [NOTICE](NOTICE) 中;仓库本身不包含轨迹数据。
