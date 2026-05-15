# DeepResearch-Agent 项目梳理

## 1. 一句话概括

DeepResearch-Agent 是一个自研多 Agent 深度研究流水线：从用户问题出发，自动规划任务、执行研究、沉淀证据、压缩上下文、处理证据冲突、生成报告、评分，并可做红蓝对抗式修复和本地 benchmark 评测。

## 2. 当前版本完成到哪里

当前项目已经从最初的线性多 Agent MVP，推进到包含以下能力的完整实验系统：

- 多 Agent 线性 MVP
- DAG 调度和 9 状态状态机
- 动态 replan 和三级降级
- OpenAI-compatible / DeepSeek / vLLM / MiMo 等真实 LLM 后端
- SQLite + numpy 共享记忆
- L1 / L2 / L3 上下文压缩
- Evidence Conflict Detection and Resolution
- Red-Blue Adversarial Repair
- ResearchBench Evaluation System
- Real Web Search + Citation Validation
- Tavily / Exa / Serper / Brave 搜索后端适配
- Xiaomi MiMo API 最小连通性测试
- MiMo structured-output repair / schema coercion / partial salvage hotfix

目前重点不是继续堆新功能，而是稳定真实模型输出质量、清理项目产物、理解主流程。

## 3. 完整执行流程

典型 `run_demo.py` 流程：

```text
User Question
-> PlannerAgent 生成研究子任务
-> TaskDAG / DAGTaskScheduler 调度任务
-> ResearcherAgent 生成 evidence，或可选 web search grounded evidence
-> MemoryStore / SQLiteMemoryStore 保存 evidence
-> optional ContextCompressionPipeline 筛选 evidence
-> optional EvidenceConflictDetector / Resolver 检测和消解矛盾
-> WriterAgent 写 Markdown 报告
-> Report Quality Check 修复基本报告格式
-> JudgeAgent 评分
-> optional RedAgent / BlueAgent 多轮审查和修复
-> 输出 outputs/demo_report.md 和 outputs/demo_trace.json
```

## 4. 每个 Agent 的作用

### PlannerAgent

负责把用户问题拆成 3-5 个研究子任务。它输出结构化 task JSON。因为真实模型可能输出非标准 JSON，目前已经支持 direct parse、lenient parse、schema coercion、LLM repair retry 和 fallback planner。

### ResearcherAgent

负责对每个子任务生成 evidence。默认可以直接让 LLM 生成 evidence；启用 web search 后，可以通过 SearchProvider、WebFetcher 和 CitationValidator 生成 grounded evidence。它也支持从非标准 evidence JSON、字段变体和截断 JSON 中恢复 Evidence。

### WriterAgent

负责基于筛选后的 evidence 写 Markdown 报告。它会结合 conflict metadata 和 evidence priority，尽量避免过度绝对化表达。

### JudgeAgent

负责从 factuality、coverage、reasoning_depth、citation_quality、clarity、overall 等维度给报告打分。真实模型输出不稳定时，它会尝试自然语言分数抽取、schema coercion、LLM repair retry，最后才使用 rule-based fallback judge。

### RedAgent

负责作为严格审稿人攻击报告，找事实性、覆盖度、逻辑、引用、证据使用、矛盾处理、结构完整性等问题。

### BlueAgent

负责根据 RedAgent 的 critique 修复报告，执行 ADD、DELETE、MODIFY、VERIFY 等动作。不能编造新 citation，只能基于已有 evidence 修复。

## 5. 每个核心模块的作用

### DAG Scheduler

负责把 PlannerAgent 生成的任务放进 DAG 中，根据依赖关系和并发限制调度 ResearcherAgent。支持超时、失败、重试、降级和 replan。

### 9 状态状态机

每个任务都有明确状态：

- `PENDING`
- `READY`
- `RUNNING`
- `SUCCEEDED`
- `FAILED`
- `TIMEOUT`
- `DEGRADED`
- `REPLANNED`
- `CANCELLED`

状态机保证任务生命周期可追踪、可调试。

### Memory

Memory 是 evidence 共享层。默认 `memory` 是内存存储；`sqlite` 后端会把 evidence、metadata、content hash 和 source quality 持久化。numpy hashing vector index 支持本地相似检索。

### Context Compression

上下文压缩用于减少 WriterAgent 输入。当前是本地轻量版：

- L1：hashing embedding 粗筛
- L2：TextRank-style 图排序
- L3：保留原始 evidence 内容

### Conflict Detection

证据矛盾检测在 WriterAgent 前执行。它识别 duplicate、near_duplicate、反义词冲突、数值方向冲突、语义对立、source quality conflict 等，并通过 drop、downweight、mark_for_writer 等策略处理。

### Red-Blue Repair

Red-Blue 是报告级审查和修复闭环。RedAgent 找问题，BlueAgent 修复，再由 JudgeAgent 复评。系统可以在分数收敛、无高严重问题或达到最大轮数时停止。

### Web Search

Web Search 是可选 grounded research 路径。它支持 MockSearchProvider 和 Tavily / Exa / Serper / Brave / web 等真实搜索后端。默认关闭，测试全部使用 mock search。

### Citation Validation

CitationValidator 当前是 token-overlap proxy，不是真实事实验证。它判断 source_url 是否存在、页面/snippet 是否可达、内容是否和 evidence claim 有重叠。

### ResearchBench Evaluation

ResearchBench 是本地 benchmark 和实验流水线。它支持 JSONL benchmark、rule metrics、Judge metrics、bootstrap CI、Cohen's d、results JSONL/CSV/Markdown 导出和 run comparison。

## 6. 当前 MiMo 接入情况

当前 MiMo 已经可以作为 `openai-compatible` backend 运行：

- API 连通性测试已完成。
- `run_demo.py` 可以进入 Agent 流程并生成报告和 trace。
- MiMo 的主要问题不是连通性，而是结构化 JSON 输出不稳定。
- 已经做了：
  - JSON repair retry
  - lenient JSON parse
  - schema coercion
  - partial JSON salvage
  - JudgeAgent fallback
  - Researcher fallback content 非空保护

下一步建议做 MiMo output quality gate：针对 MiMo 单独评估 Planner / Researcher / Judge 的结构化输出成功率，再决定是否做 MiMo prompt profile。

## 7. 当前最重要的调试 trace 字段

### planner_stats

看 Planner 是否成功：

- `json_parse_success`
- `repair_attempted`
- `repair_success`
- `schema_coercion_success`
- `partial_repair_used`
- `fallback_used`
- `subtask_count`
- `error`

### collected_evidences.metadata

看每条 evidence 是否来自正常 JSON、repair、schema coercion 或 fallback：

- `json_parse_success`
- `repair_attempted`
- `repair_success`
- `schema_coercion_success`
- `partial_json_salvaged`
- `fallback_parse`
- `grounded`
- `citation_validation_status`

### judge_stats

看 Judge 是否稳定：

- `json_parse_success`
- `extracted_from_text`
- `repair_attempted`
- `repair_success`
- `schema_coercion_success`
- `fallback_used`
- `normalized`
- `error`

### memory_stats

看 evidence 存储和检索情况：

- `backend`
- `inserted_evidence_count`
- `duplicate_evidence_count`
- `total_evidence_count`
- `retrieved_evidence_count`
- `source_quality_summary`

### compression_stats

看上下文压缩是否启用以及压缩比例：

- `enabled`
- `l1_input_count`
- `l1_selected_count`
- `l2_selected_count`
- `final_selected_count`
- `compression_ratio`
- `selected_evidence_ids`

### conflict_stats

看证据冲突检测结果：

- `enabled`
- `conflict_count`
- `duplicate_count`
- `near_duplicate_count`
- `numeric_direction_conflict_count`
- `semantic_opposition_count`
- `dropped_evidence_ids`
- `marked_conflict_evidence_ids`

### red_blue_stats

看红蓝修复是否有效：

- `enabled`
- `rounds_completed`
- `initial_overall_score`
- `final_overall_score`
- `score_delta`
- `total_red_issues`
- `total_patches_applied`
- `stopped_reason`

### search_stats

看搜索和 citation validation：

- `enabled`
- `provider`
- `query_count`
- `result_count`
- `validated_citation_count`
- `supported_count`
- `unsupported_count`
- `provider_errors`
- `fallback_used`

## 8. 当前建议

短期建议：

- 暂停新增大功能。
- 先清理 `.pytest_tmp*`、`outputs/`、`data/`、`eval_outputs/` 等运行产物。
- 先完整理解 `run_demo.py` 主流程和 trace 字段。
- 再继续优化 MiMo prompt profile。
- 不要把 API Key 写入代码、README、trace、日志或截图。

下一步最值得做的不是继续扩 Agent，而是建立 MiMo output quality gate：

- 单独测 Planner JSON 成功率。
- 单独测 Researcher evidence JSON 成功率。
- 单独测 Judge score JSON 成功率。
- 记录 direct parse、schema coercion、repair retry、fallback 的比例。
- 基于结果再决定是否需要 MiMo-specific prompts。
