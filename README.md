# DeepResearch-Agent

DeepResearch-Agent 是一个自研的多 Agent 深度研究系统。它的目标不是简单调用一次大模型生成报告，而是把“研究问题拆解、证据检索、证据保存、上下文筛选、冲突检测、报告生成、质量检查、评审修复、实验评测”串成一条可观察、可测试、可扩展的研究流水线。

这个项目目前没有使用 LangChain、LangGraph、AutoGen、CrewAI 等外部 Agent 编排框架，核心调度、状态机、记忆、压缩、冲突检测、评测和 search provider 都是项目内自研实现。

当前阶段：**v1.0 Research Grounding Quality Upgrade**。项目的业务场景已经收敛为：**面向 AI / LLM / 通信科研调研的工具增强型多 Agent 深度研究助手**。下一阶段重点不是继续扩张 Agent 数量，而是提升 retrieval quality、citation grounding 和 report references。

---

## 1. 项目一句话概括

DeepResearch-Agent 是一个面向“深度研究报告生成”的多 Agent 系统：

```text
用户问题
-> PlannerAgent 拆解任务
-> DAG Scheduler 调度 ResearcherAgent
-> SearchProvider 检索真实来源
-> CitationValidator 做引用支持度检查
-> Memory 保存 evidence
-> Compression / Conflict / Red-Blue 做质量控制
-> WriterAgent 写报告
-> JudgeAgent 评分
-> 输出 report + trace
```

它适合作为简历或研究项目展示，因为它覆盖了 Agent 系统里比较完整的一组工程能力：

- 多 Agent 协作
- DAG 调度
- 状态机
- 异步并发
- 失败重试与降级
- 共享记忆
- 上下文压缩
- 证据冲突检测
- 真实检索 grounding
- 引用验证 proxy
- 报告质量检查
- Red-Blue 对抗修复
- 本地 benchmark 评测体系

---

## 2. 当前完整执行流程

当前主流程如下：

```text
1. 用户输入 question
2. PlannerAgent 生成 3-5 个 research subtasks
3. TaskDAG 保存任务和依赖关系
4. DAGTaskScheduler 按依赖和并发限制调度任务
5. ResearcherAgent 为每个 task 收集 evidence
   5.1 如果未启用 web search：使用 LLM 生成 evidence
   5.2 如果启用 web search：走 SearchProvider + CitationValidator
6. MemoryStore 保存 evidence
7. 可选：ContextCompressionPipeline 筛选 evidence
8. 可选：EvidenceConflictDetector / Resolver 清理重复、冲突、低质量 evidence
9. WriterAgent 基于 evidence 写 Markdown 报告
10. Report Quality Check 检查报告是否截断、缺 section、格式不完整
11. JudgeAgent 给报告打分
12. 可选：RedAgent / BlueAgent 做多轮审查与修复
13. 输出：
    - outputs/demo_report.md
    - outputs/demo_trace.json
```

简化图：

```text
Question
  |
  v
PlannerAgent
  |
  v
TaskDAG + DAGTaskScheduler
  |
  v
ResearcherAgent
  |
  +-- LLM evidence
  |
  +-- Search grounding
      -> SearchQueryGenerator
      -> SearchProvider
      -> WebFetcher
      -> CitationValidator
      -> Grounded Evidence
  |
  v
Memory
  |
  v
Compression -> Conflict Detection
  |
  v
WriterAgent
  |
  v
Report Quality Check
  |
  v
JudgeAgent
  |
  v
Red-Blue Repair
  |
  v
Report + Trace
```

---

## 3. 快速开始

### 3.1 安装依赖

建议在虚拟环境中安装：

```powershell
python -m pip install -e ".[dev]"
```

### 3.2 最小 mock 链路测试

这个命令不需要 API key、不需要网络，适合确认项目基本可运行：

```powershell
python scripts/run_demo.py `
  --question "test long-context evaluation" `
  --backend mock `
  --mode dag
```

输出文件：

```text
outputs/demo_report.md
outputs/demo_trace.json
```

### 3.3 运行全量测试

```powershell
pytest --basetemp=.pytest_tmp
```

当前测试全部使用 mock、monkeypatch 或本地文件，不依赖真实 API key。

---

## 4. MiMo API 接入

MiMo 已经可以作为 `openai-compatible` backend 使用。建议先做最小 API 连通性测试，再跑完整 Agent 链路。

### 4.1 设置环境变量

PowerShell 示例：

```powershell
$env:XIAOMI_MIMO_API_KEY="your_key"
$env:XIAOMI_MIMO_API_BASE="https://token-plan-cn.xiaomimimo.com/v1"
$env:XIAOMI_MIMO_MODEL="your_console_model_name"
```

注意：

- 模型名必须以小米控制台实际显示为准。
- 不要把真实 API Key 写进代码、README、trace、日志、截图或聊天记录。
- 如果控制台给的是专属 Base URL，应优先使用专属 Base URL。

### 4.2 最小 API 连通性测试

```powershell
python scripts/test_mimo_api.py
```

常见结果：

- `200`：API 连通，模型名和 key 基本可用。
- `401 / 403`：API key 错误、过期或权限不足。
- `400 Not supported model`：模型名不对，以控制台显示为准。
- `402 Insufficient account balance`：账户额度或套餐权限问题。
- `404`：Base URL 或 endpoint 不对。
- timeout：网络或代理问题。

详细说明见：

```text
docs/mimo_api_test.md
```

### 4.3 MiMo 最小 Agent demo

先不要打开 search、compression、red-blue，先确认 MiMo 能跑通 Agent 主链路：

```powershell
python scripts/run_demo.py `
  --question "What are the main challenges of long-context LLM evaluation?" `
  --backend openai-compatible `
  --mode dag `
  --max-concurrency 1 `
  --api-base $env:XIAOMI_MIMO_API_BASE `
  --api-key-env XIAOMI_MIMO_API_KEY `
  --model $env:XIAOMI_MIMO_MODEL `
  --max-tokens 2048 `
  --request-timeout 240 `
  --global-timeout-seconds 720
```

### 4.4 MiMo + arXiv 论文 grounding

如果最小 Agent demo 正常，再启用 arXiv 检索：

```powershell
python scripts/run_demo.py `
  --question "What are the main challenges of long-context LLM evaluation?" `
  --backend openai-compatible `
  --mode dag `
  --max-concurrency 1 `
  --api-base $env:XIAOMI_MIMO_API_BASE `
  --api-key-env XIAOMI_MIMO_API_KEY `
  --model $env:XIAOMI_MIMO_MODEL `
  --max-tokens 2048 `
  --request-timeout 240 `
  --global-timeout-seconds 720 `
  --enable-web-search `
  --search-provider arxiv `
  --max-search-queries 1 `
  --search-top-k 2
```

这个命令会让 ResearcherAgent 使用 arXiv 论文作为 grounded evidence，而不是完全依赖 LLM 内部知识。

### 4.5 MiMo + arXiv 语义通信调研 demo

当前 v1.0 推荐用 AI / LLM / 通信科研问题来验证 grounding 质量，例如：

```powershell
python scripts/run_demo.py `
  --question "What are recent methods for task-oriented semantic communication with LLMs?" `
  --backend openai-compatible `
  --mode dag `
  --max-concurrency 1 `
  --api-base $env:XIAOMI_MIMO_API_BASE `
  --api-key-env XIAOMI_MIMO_API_KEY `
  --model $env:XIAOMI_MIMO_MODEL `
  --max-tokens 2048 `
  --request-timeout 240 `
  --global-timeout-seconds 720 `
  --enable-web-search `
  --search-provider arxiv `
  --max-search-queries 1 `
  --search-top-k 2
```

建议对比两种结果：

```text
纯 LLM evidence：不加 --enable-web-search
grounded evidence：加 --enable-web-search --search-provider arxiv
```

对比重点不是报告是否更长，而是：

- evidence 是否有真实 source_url
- References 是否列出 evidence id、title、source_url、citation status
- `search_stats.provider_errors` 是否为空
- `collected_evidences[*].metadata.retrieval_relevance_score` 是否合理
- `citation_validation_status` 是否比以前更保守

---

## 5. 当前 Agent 说明

### PlannerAgent

负责把用户问题拆成多个研究子任务。理想输出是结构化 JSON。

对于 long-context LLM evaluation 这类问题，Planner 应该拆出：

- benchmark
- metrics
- context retention
- multi-hop reasoning
- lost-in-the-middle
- faithfulness
- latency
- cost
- scalability

为了兼容 MiMo 等 OpenAI-compatible 模型偶尔不稳定输出 JSON，PlannerAgent 已经支持：

```text
direct JSON parse
-> lenient JSON parse
-> schema coercion
-> partial JSON salvage
-> LLM repair retry
-> local fallback
```

### ResearcherAgent

负责执行每个 research task，并生成 evidence。

两种路径：

```text
纯 LLM 路径：
task -> LLM -> evidence

grounded research 路径：
task -> search query -> SearchProvider -> fetched document -> citation validation -> grounded evidence
```

如果启用 `--enable-web-search`，ResearcherAgent 会优先使用 grounded research。搜索失败时会 fallback 到原始 LLM evidence 流程。

### WriterAgent

负责把 evidence 写成 Markdown 报告。

当前要求报告包含：

- `## Abstract`
- `## Key Findings`
- `## Evidence Summary`
- `## Limitations`
- `## Conclusion`

WriterAgent 会按 task 分组选择 top-k evidence，并且优先避免使用 degraded fallback evidence。

### JudgeAgent

负责给报告评分。输出字段：

- `factuality`
- `coverage`
- `reasoning_depth`
- `citation_quality`
- `clarity`
- `overall`
- `comments`

JudgeAgent 也支持 JSON 修复、自然语言分数抽取和 fallback judge。

### RedAgent

负责严格审查报告，找出：

- factuality 问题
- coverage 问题
- reasoning_depth 问题
- citation_quality 问题
- evidence_mismatch 问题
- contradiction 问题
- unsupported_claim 问题
- report_structure 问题
- clarity 问题
- conflict_handling 问题

### BlueAgent

负责根据 RedAgent 的 critique 修复报告。

支持的修复动作：

- `ADD`
- `DELETE`
- `MODIFY`
- `VERIFY`
- `NOOP`

---

## 6. 核心模块说明

### DAG Scheduler

`DAGTaskScheduler` 是调度核心，负责：

- 根据依赖关系找到 ready tasks
- 控制并发数
- 设置 task timeout
- 调用 ResearcherAgent
- 处理失败、重试、超时、降级
- 记录 task state transition

### 9 状态状态机

任务状态包括：

- `PENDING`
- `READY`
- `RUNNING`
- `SUCCEEDED`
- `FAILED`
- `TIMEOUT`
- `DEGRADED`
- `REPLANNED`
- `CANCELLED`

状态机保证任务状态流转合法，也让 trace 更容易分析。

### Memory

Memory 是 evidence 共享层。

支持：

- in-memory backend
- SQLite backend
- content hash 去重
- source quality label
- numpy vector index
- hashing embedding provider

SQLite 示例：

```powershell
python scripts/run_demo.py `
  --question "test long-context evaluation" `
  --backend mock `
  --mode dag `
  --memory-backend sqlite `
  --clear-memory
```

### Context Compression

用于 evidence 很多时筛选上下文。

当前实现：

- L1：local hashing embedding 粗筛
- L2：TextRank 风格图排序
- L3：保留原始 evidence，不做摘要改写

启用方式：

```powershell
python scripts/run_demo.py `
  --question "test long-context evaluation" `
  --backend mock `
  --mode dag `
  --enable-context-compression
```

### Evidence Conflict Detection

用于写报告前清理 evidence。

可以检测：

- duplicate
- near_duplicate
- antonym_contradiction
- numeric_direction_conflict
- semantic_opposition
- source_quality_conflict
- off_topic

启用方式：

```powershell
python scripts/run_demo.py `
  --question "test long-context evaluation" `
  --backend mock `
  --mode dag `
  --enable-conflict-detection
```

### Red-Blue Repair

用于报告生成后的多轮审查与修复。

启用方式：

```powershell
python scripts/run_demo.py `
  --question "test long-context evaluation" `
  --backend mock `
  --mode dag `
  --enable-red-blue
```

建议基础链路稳定后再打开 Red-Blue，否则排查问题会变复杂。

---

## 7. SearchProvider 与 Citation Validation

当前支持的 SearchProvider：

| Provider | 说明 | 是否需要网络 | 是否需要 API Key |
|---|---|---:|---:|
| `mock` | 本地假搜索，测试用 | 否 | 否 |
| `arxiv` | arXiv Atom API 论文检索 | 是 | 否 |
| `paper` | 当前是 arXiv 的语义别名 | 是 | 否 |
| `tavily` | Agent/RAG 友好的真实搜索 | 是 | 是 |
| `exa` | 语义搜索、研究型搜索 | 是 | 是 |
| `serper` | Google SERP 风格搜索 | 是 | 是 |
| `brave` | Brave 独立 web index | 是 | 是 |
| `web` | 通用 HTTP search adapter | 是 | 是 |

真实搜索 API Key 建议用环境变量：

```powershell
$env:TAVILY_API_KEY="your_key"
$env:EXA_API_KEY="your_key"
$env:SERPER_API_KEY="your_key"
$env:BRAVE_API_KEY="your_key"
```

示例：

```powershell
python scripts/run_demo.py `
  --question "long context LLM evaluation benchmark" `
  --backend mock `
  --mode dag `
  --enable-web-search `
  --search-provider tavily `
  --search-api-key-env TAVILY_API_KEY
```

### Citation Validation 当前限制

当前 `CitationValidator` 是轻量 proxy，不是严格事实验证。

它主要使用：

```text
title token overlap
content token overlap
source snippet overlap
```

因此：

- `supported` 表示 evidence 和 source 文本相关度较高。
- `supported` 不等于论文严格证明了该 claim。
- 对领域不匹配但关键词重合的论文，当前 validator 可能过于宽松。

这是后续最值得增强的地方之一。

v1.0 已经把 CitationValidator 调得更保守：

- `mock://degraded` 不再被当作高质量 citation。
- 只有 source title / snippet / content 与 evidence claim 有较强重合时才判为 `supported`。
- 弱重合或模型泛化出来的 claim 更容易被判为 `partially_supported` 或 `unsupported`。
- arXiv / paper evidence 会保留 `retrieval_relevance_score`、`retrieval_relevance_reason` 和 `reranked_by_relevance`，方便在 trace 中排查检索质量。
- arXiv provider 会对 429 / 5xx / timeout 做轻量退避重试，并缓存成功查询结果，减少重复请求导致的限流。
- 如果检索失败后退回纯 LLM evidence，Researcher 会给 evidence id 加 task 前缀，降低无来源 evidence 置信度，并清理无来源内容中的具体百分比等高风险数字 claim。

---

## 8. ResearchBench Evaluation

项目内置了一个本地评测体系。

Benchmark 文件：

```text
benchmarks/researchbench.jsonl
benchmarks/hotpotqa_style.jsonl
```

评测能力：

- rule metrics
- Judge metrics
- bootstrap 95% CI
- Cohen's d
- JSONL / CSV / Markdown summary 导出
- 多配置实验矩阵

检查 benchmark：

```powershell
python scripts/check_benchmark.py --benchmark benchmarks/researchbench.jsonl
```

运行小规模评测：

```powershell
python scripts/run_benchmark.py `
  --benchmark benchmarks/researchbench.jsonl `
  --backend mock `
  --limit 5 `
  --mode dag
```

运行实验矩阵：

```powershell
python scripts/run_experiment_matrix.py `
  --benchmark benchmarks/researchbench.jsonl `
  --limit 5
```

比较两个 run：

```powershell
python scripts/compare_runs.py `
  --baseline eval_outputs/run_a/results.jsonl `
  --treatment eval_outputs/run_b/results.jsonl `
  --metric judge_overall
```

---

## 9. Streamlit UI

项目包含一个轻量 Streamlit Demo UI。

运行方式：

```powershell
streamlit run app/streamlit_app.py
```

UI 支持查看：

- report
- trace summary
- task states
- execution steps
- memory stats
- compression stats
- conflict stats
- red-blue stats
- search and citation stats
- evaluation summary

UI 不建议直接跑大型 benchmark。大型实验请使用 CLI。

---

## 10. 如何看 trace

每次运行后重点看：

```text
outputs/demo_trace.json
outputs/demo_report.md
```

最重要的 trace 字段：

| 字段 | 用途 |
|---|---|
| `planner_stats` | Planner 是否成功 JSON 输出，是否 fallback |
| `planned_subtasks` | 子任务是否围绕原问题，是否跑偏 |
| `collected_evidences` | evidence 的 title/content/source/metadata |
| `search_stats` | 搜索是否启用、provider、query/result 数量、provider errors |
| `memory_stats` | evidence 写入数量、source quality 分布 |
| `compression_stats` | 是否压缩，保留了哪些 evidence |
| `conflict_stats` | 是否发现重复、冲突、低质量 evidence |
| `red_blue_stats` | Red-Blue 是否启用、修复轮次、分数变化 |
| `judge_stats` | Judge 是否 JSON 成功，是否 fallback |
| `final_judge_score` | 最终评分 |
| `task_duration_seconds` | 每个任务耗时 |
| `degraded_tasks` | 是否有降级任务 |
| `degradation_records` | 具体降级原因 |

### 当前最新 MiMo + arXiv 结果说明

最近一次链路显示：

- Planner 正常：`json_parse_success=true`
- Judge 正常：`json_parse_success=true`
- DAG 正常：所有 task 都 `SUCCEEDED`
- arXiv 检索正常：`provider_errors=[]`
- memory 中 evidence 都有真实 URL
- report 结构完整

当前主要问题已经不是“链路跑不通”，而是“科研证据质量还要增强”：

- arXiv 可能召回领域相近但不完全匹配的论文。
- CitationValidator 目前只做 token overlap，可能过于宽松。
- Writer 有时输出数字引用 `[1]`，但没有完整 References 映射。
- `grounded_fallback=true` 的 evidence 应该比正常 grounded evidence 权重更低。

---

## 11. 推荐调试顺序

不要一开始就把所有功能都打开。建议按下面顺序：

### 第一步：mock 最小链路

```powershell
python scripts/run_demo.py `
  --question "test long-context evaluation" `
  --backend mock `
  --mode dag
```

### 第二步：MiMo API 连通性

```powershell
python scripts/test_mimo_api.py
```

### 第三步：MiMo Agent 最小链路

```powershell
python scripts/run_demo.py `
  --question "What are the main challenges of long-context LLM evaluation?" `
  --backend openai-compatible `
  --mode dag `
  --max-concurrency 1 `
  --api-base $env:XIAOMI_MIMO_API_BASE `
  --api-key-env XIAOMI_MIMO_API_KEY `
  --model $env:XIAOMI_MIMO_MODEL `
  --max-tokens 2048 `
  --request-timeout 240 `
  --global-timeout-seconds 720
```

### 第四步：MiMo + arXiv grounding

```powershell
python scripts/run_demo.py `
  --question "What are the main challenges of long-context LLM evaluation?" `
  --backend openai-compatible `
  --mode dag `
  --max-concurrency 1 `
  --api-base $env:XIAOMI_MIMO_API_BASE `
  --api-key-env XIAOMI_MIMO_API_KEY `
  --model $env:XIAOMI_MIMO_MODEL `
  --max-tokens 2048 `
  --request-timeout 240 `
  --global-timeout-seconds 720 `
  --enable-web-search `
  --search-provider arxiv `
  --max-search-queries 1 `
  --search-top-k 2
```

### 第五步：逐步打开增强模块

```text
--memory-backend sqlite --clear-memory
--enable-context-compression
--enable-conflict-detection
--enable-red-blue
```

如果一步全开，排查问题会很困难，因为你很难判断问题来自 LLM、search、memory、compression、conflict 还是 red-blue。

---

## 12. 项目目录结构

```text
DeepResearch-Agent/
|-- app/
|   `-- streamlit_app.py
|-- benchmarks/
|   |-- researchbench.jsonl
|   `-- hotpotqa_style.jsonl
|-- docs/
|-- scripts/
|   |-- run_demo.py
|   |-- test_mimo_api.py
|   |-- run_benchmark.py
|   |-- run_experiment_matrix.py
|   |-- summarize_results.py
|   |-- compare_runs.py
|   |-- export_eval_report.py
|   `-- clean_workspace.py
|-- src/deepresearch_agent/
|   |-- agents/
|   |-- engine/
|   |-- memory/
|   |-- compression/
|   |-- conflict/
|   |-- red_blue/
|   |-- search/
|   |-- evaluation/
|   |-- report/
|   |-- ui/
|   |-- llm/
|   `-- utils/
|-- tests/
|-- README.md
|-- pyproject.toml
`-- .gitignore
```

---

## 13. 清理本地运行产物

项目提供安全清理脚本，默认 dry-run，不会真的删除：

```powershell
python scripts/clean_workspace.py
```

真正删除默认缓存：

```powershell
python scripts/clean_workspace.py --yes
```

包含 outputs / data / eval outputs：

```powershell
python scripts/clean_workspace.py --include-all --yes
```

默认或可选清理：

- `.pytest_cache/`
- `.pytest_tmp*/`
- `__pycache__/`
- `*.pyc`
- `outputs/`
- `outputs_*/`
- `eval_outputs/`
- `data/`
- `*.sqlite`
- `*.db`
- `*.npz`
- `*.npy`

脚本不会删除源码、测试、文档、配置、`.git`、`.venv` 等关键目录。

---

## 14. 安全注意事项

不要提交：

- `.env`
- API Key
- `outputs/`
- `outputs_*/`
- `eval_outputs/`
- `data/`
- SQLite 数据库
- numpy index
- 日志文件

API Key 推荐只通过环境变量传入：

```powershell
$env:MY_API_KEY="your_key"
```

然后 CLI 使用：

```powershell
--api-key-env MY_API_KEY
```

---

## 15. 当前不建议做的事

当前项目已经比较复杂，不建议继续盲目加大模块。

不建议：

- 重写 DAG Scheduler。
- 把项目整体迁移到 LangChain / LangGraph。
- 删除 mock backend。
- 一次接入太多真实 API。
- 继续增加更多 Agent。
- 继续堆复杂 quality gate。
- 让测试依赖真实网络或真实 API Key。

当前真正值得做的是提高研究质量，而不是继续扩大架构。

---

## 16. 下一步建议

v1.0 已经完成的小步优化：

```text
1. arXiv / paper provider 增加轻量相关性过滤和 reranking
2. CitationValidator 更保守
3. Writer 强制补 Evidence Summary evidence id 和 References
4. grounded_fallback / degraded evidence 降权
```

下一步仍然推荐继续沿着这个方向小步优化：

```text
1. 用 ResearchBench 对比 pure LLM evidence vs grounded evidence
2. 继续增强 paper category penalty
3. 增加更细的 citation reranking
4. 后续再考虑 LLM-as-verifier
```

这几项比继续新增 Agent 更有价值，因为当前主链路已经跑通，短板主要在：

```text
检索更准
引用更可信
报告更科研化
评测更可重复
```

---

## 17. 当前项目状态总结

DeepResearch-Agent 现在已经从“能跑的多 Agent demo”进入了“真实科研助手雏形”阶段。

已经解决的问题：

- 多 Agent 流程能跑通。
- DAG 调度和状态机可观察。
- mock backend 可稳定测试。
- MiMo 可作为 openai-compatible backend 接入。
- arXiv grounding 可以提供真实论文来源。
- trace 能详细解释每一步发生了什么。
- 全量测试不依赖真实 API。

当前仍需增强的问题：

- arXiv 检索结果相关性还需要更严格。
- Citation validation 还是 proxy，不是真正事实验证。
- Writer 引用格式需要更规范。
- grounded fallback evidence 应该降权。
- Red-Blue 和 Evaluation 更适合在基础检索质量稳定后使用。

一句话：**架构已经成立，下一阶段应该集中提升 retrieval quality 和 citation grounding，而不是继续扩张 Agent 数量。**
