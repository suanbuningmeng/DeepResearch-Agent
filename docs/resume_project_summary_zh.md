# DeepResearch-Agent 简历版项目说明

## 一句话定位

DeepResearch-Agent 是一个面向 AI / LLM / 通信科研调研的工具增强型多 Agent 深度研究助手，支持问题拆解、DAG 调度、真实论文检索、证据 grounding、引用验证、报告生成和可观测 trace。

## 简历推荐写法

**DeepResearch-Agent：多 Agent 科研调研与论文 Grounding 系统**

- 自研 Planner / Researcher / Writer / Judge 多 Agent 流水线，使用 asyncio + Semaphore 实现 DAG 任务调度、任务状态流转、失败降级和 trace 可观测。
- 设计 SearchProvider 抽象，接入 arXiv、Mock Search 及 Tavily / Exa / Serper / Brave 等可插拔检索后端，将 LLM 生成证据升级为真实论文 grounding evidence。
- 实现 CitationValidator、source quality scoring 和 report references 机制，在报告中显式保留 evidence id、source URL 与 citation support 状态，降低无来源幻觉风险。
- 构建 Memory、Context Compression、Conflict Detection、Red-Blue Repair 和 ResearchBench Evaluation 等可选增强模块，支持从最小链路到完整研究流水线的扩展。
- 接入 MiMo / Qwen / DeepSeek / vLLM 等 OpenAI-compatible 后端，并增加结构化 JSON 解析、repair retry、schema coercion 和 LLM usage trace，提升真实模型运行稳定性。

## 面试时建议主讲的核心链路

不要从所有模块开始讲，建议只讲这一条主线：

```text
Question
-> PlannerAgent 拆解研究任务
-> DAGTaskScheduler 并发调度
-> ResearcherAgent 调用 arXiv / SearchProvider 检索论文
-> CitationValidator 检查证据和来源支持度
-> WriterAgent 生成带 References 的报告
-> JudgeAgent 评分
-> demo_trace.json 记录全过程
```

这条链路最适合展示工程能力，因为它同时体现：

- Agent workflow 设计
- 异步任务调度
- 真实检索 grounding
- 证据质量控制
- 报告生成
- 可观测性和可测试性

## 简历版稳定运行命令

推荐使用 `--preset resume` 跑简历展示链路。它会默认启用：

- DAG 模式
- 单并发，便于稳定复现
- arXiv grounding
- `--disable-thinking`
- 较保守的搜索规模
- 关闭 Red-Blue / Conflict / Compression 等高级可选模块

```powershell
python scripts/run_demo.py `
  --preset resume `
  --question "What are recent methods for task-oriented semantic communication with LLMs?" `
  --backend openai-compatible `
  --api-base $env:XIAOMI_MIMO_API_BASE `
  --api-key-env XIAOMI_MIMO_API_KEY `
  --model $env:XIAOMI_MIMO_MODEL
```

如果想提高检索覆盖面，可以显式覆盖：

```powershell
python scripts/run_demo.py `
  --preset resume `
  --question "What are recent methods for task-oriented semantic communication with LLMs?" `
  --backend openai-compatible `
  --api-base $env:XIAOMI_MIMO_API_BASE `
  --api-key-env XIAOMI_MIMO_API_KEY `
  --model $env:XIAOMI_MIMO_MODEL `
  --max-search-queries 2 `
  --search-top-k 2
```

## 不建议在简历主线里展开的内容

这些模块可以作为“扩展能力”提一句，不建议一开始全部展开：

- Red-Blue Repair
- Conflict Detection
- Context Compression
- ResearchBench Evaluation
- 多个生产搜索 provider 的全部细节
- MiMo JSON repair 的所有 fallback 细节

面试主线应该保持清楚：**从问题到论文证据，再到带引用报告。**

## 当前项目优势

- 没有依赖 LangChain / LangGraph，核心调度、状态机、memory、compression、search provider 和 evaluation 都是自研实现。
- 不只是 prompt demo，而是有任务状态、失败恢复、trace、测试和可复现实验脚本。
- 能展示真实工程问题：模型 JSON 不稳定、检索跑偏、citation 不可靠、报告截断、API 兼容差异。

## 当前项目限制

- Citation validation 仍是 token overlap proxy，不是真实事实验证。
- arXiv 检索覆盖面取决于 query 和 arXiv 数据本身。
- MiMo 等 openai-compatible 模型仍可能产生较多 reasoning tokens。
- Red-Blue 和 ResearchBench 更适合作为扩展能力，不建议默认打开。

## 下一步最值得优化

1. 提高 arXiv / paper 检索覆盖面和去重能力。
2. 增强 citation reranking，让不同论文来源覆盖不同 evidence。
3. 将简历展示 demo 固定在 `--preset resume`，避免每次手动拼长命令。
4. 保持高级模块可选，不继续堆 Agent 数量。
