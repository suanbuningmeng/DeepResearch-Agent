# DeepResearch-Agent MVP 代码导读

这份文档用于解释当前第一阶段 MVP 的代码结构和执行流程。当前版本的重点不是“真实研究能力”，而是把多 Agent 深度研究系统的最小闭环跑通：规划、研究、记忆、写作、评审、输出报告和 trace。

## 1. 项目目录结构说明

```text
DeepResearch-Agent/
├── README.md
├── pyproject.toml
├── .env.example
├── scripts/
│   └── run_demo.py
├── src/
│   └── deepresearch_agent/
│       ├── __init__.py
│       ├── schemas.py
│       ├── llm/
│       │   ├── base.py
│       │   └── mock_client.py
│       ├── agents/
│       │   ├── planner.py
│       │   ├── researcher.py
│       │   ├── writer.py
│       │   └── judge.py
│       ├── memory/
│       │   └── store.py
│       └── report/
│           └── formatter.py
└── tests/
```

主要分层如下：

- `schemas.py`：定义系统中流转的数据结构，例如任务、证据、评分和报告。
- `llm/`：定义 LLM 抽象接口和 mock 实现。当前阶段不调用真实 API。
- `agents/`：定义四类 Agent：Planner、Researcher、Writer、Judge。
- `memory/`：定义证据存储。当前是内存列表。
- `report/`：定义报告格式化函数。
- `scripts/run_demo.py`：把所有模块串起来，形成可运行 demo。
- `tests/`：验证 mock LLM、Agent 串联和 demo 输出。

## 2. schemas.py 中的数据结构

`TaskState` 是任务状态枚举。它提前定义了 `PENDING`、`READY`、`RUNNING`、`SUCCEEDED`、`FAILED`、`TIMEOUT`、`DEGRADED`、`REPLANNED`、`CANCELLED` 等状态。当前 MVP 还没有完整状态机，但这些状态为后续调度、重试和降级预留了语义。

`TaskNode` 表示一个研究子任务。它包含任务 ID、名称、描述、负责的 Agent 类型、依赖任务、当前状态、输入输出、错误信息、超时配置和重试次数。PlannerAgent 会生成多个 `TaskNode`，ResearcherAgent 会逐个处理它们。

`Evidence` 表示研究证据。它记录证据 ID、来源任务 ID、标题、内容、来源链接、置信度和元数据。当前证据由 MockLLM 生成，但结构上已经接近后续真实检索、论文解析或网页抓取后的统一证据格式。

`JudgeScore` 表示报告评分。它从事实性、覆盖度、推理深度、引用质量、清晰度和总体分数几个维度评价报告，并附带文字评论。

`ResearchReport` 表示最终研究报告对象。它包含原始问题、Markdown 报告正文、使用到的证据列表，以及可选的 Judge 评分。

## 3. BaseLLM 和 MockLLM 的作用

`BaseLLM` 是统一的 LLM 接口。它只定义一个异步方法 `agenerate(prompt, **kwargs)`，表示“输入 prompt，返回模型生成的字符串”。

`MockLLM` 是当前 MVP 的默认 backend。它不会调用任何真实 API，而是根据 `prompt_type` 或 prompt 内容判断请求类型，然后返回固定但合理的结果：

- planner 请求：返回包含 `subtasks` 的 JSON。
- researcher 请求：返回包含 `evidences` 的 JSON。
- writer 请求：返回 Markdown 报告。
- judge 请求：返回评分 JSON。

这样做的好处是：即使没有 API Key、没有网络、没有真实模型，整个系统也能跑通。

## 4. PlannerAgent 的输入、输出和执行逻辑

`PlannerAgent.plan(question: str) -> list[TaskNode]`

输入是用户的研究问题，例如：

```text
What are the main challenges and recent methods for long-context LLM evaluation?
```

PlannerAgent 会构造一个规划 prompt，要求 LLM 把问题拆成 3-5 个子任务。MockLLM 返回固定 JSON 后，PlannerAgent 将每个子任务转换成 `TaskNode`。输出是 `TaskNode` 列表。

当前 MVP 中，PlannerAgent 不做复杂依赖分析，也不生成 DAG，只生成简单的线性子任务集合。

## 5. ResearcherAgent 的输入、输出和执行逻辑

`ResearcherAgent.research(task: TaskNode) -> list[Evidence]`

输入是 PlannerAgent 生成的某个 `TaskNode`。ResearcherAgent 会根据任务 ID、任务名称和任务描述构造 researcher prompt，并让 MockLLM 返回证据 JSON。

随后它把 JSON 中的每条 evidence 转换成 `Evidence` 对象，并自动生成证据 ID，例如：

```text
task_1_evidence_1
task_1_evidence_2
```

输出是 `Evidence` 列表。

## 6. MemoryStore 的作用

`MemoryStore` 是当前 MVP 的简单证据仓库。它用内存中的 list 保存 `Evidence` 对象，支持：

- `add_evidence(evidence)`：保存单条证据。
- `add_evidences(evidences)`：批量保存证据。
- `list_evidences()`：读取当前所有证据。
- `clear()`：清空证据。

它的意义是把 ResearcherAgent 产生的证据集中起来，供 WriterAgent 和 JudgeAgent 使用。后续可以替换成 SQLite、向量数据库或共享记忆系统，但当前阶段只做最小可运行版本。

## 7. WriterAgent 的输入、输出和执行逻辑

`WriterAgent.write(question: str, evidences: list[Evidence]) -> str`

输入是原始问题和 MemoryStore 中收集到的证据列表。WriterAgent 会把 evidence 简要拼成文本，构造写作 prompt，并要求 LLM 生成 Markdown 报告。

当前 MockLLM 返回的报告包含这些部分：

- Title
- Abstract
- Key Findings
- Evidence Summary
- Limitations
- Conclusion

输出是 Markdown 字符串。

## 8. JudgeAgent 的输入、输出和执行逻辑

`JudgeAgent.judge(question, report, evidences) -> JudgeScore`

输入是原始问题、WriterAgent 生成的报告文本，以及证据列表。JudgeAgent 构造评分 prompt，让 MockLLM 返回评分 JSON。

返回值会被解析成 `JudgeScore`，包括事实性、覆盖度、推理深度、引用质量、清晰度、总体评分和评论。

## 9. scripts/run_demo.py 如何串起完整流程

`scripts/run_demo.py` 是当前 MVP 的主入口。运行命令示例：

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock
```

执行顺序如下：

1. 解析命令行参数。
2. 创建 `MockLLM`。
3. 创建 `MemoryStore`。
4. 创建 Planner、Researcher、Writer、Judge 四个 Agent。
5. PlannerAgent 将问题拆成多个 `TaskNode`。
6. ResearcherAgent 逐个处理子任务，生成 evidence。
7. MemoryStore 保存所有 evidence。
8. WriterAgent 根据问题和 evidence 生成 Markdown 报告。
9. JudgeAgent 对报告评分。
10. 将报告写入 `outputs/demo_report.md`。
11. 将执行过程写入 `outputs/demo_trace.json`。

这个脚本体现了当前系统的最小闭环。

## 10. tests/ 中每个测试文件在测什么

`tests/test_mock_llm.py` 验证 MockLLM 对四类 prompt 都能返回预期格式：planner 返回 subtasks，researcher 返回 evidences，writer 返回 Markdown，judge 返回评分。

`tests/test_agents.py` 验证四个 Agent 能串起来执行：先规划子任务，再收集证据，再写报告，最后评分。

`tests/test_run_demo.py` 通过子进程运行 `scripts/run_demo.py`，验证 demo 是否能生成 `demo_report.md` 和 `demo_trace.json`，并检查 trace 中的关键字段。

## 11. demo_report.md 和 demo_trace.json 的意义

`demo_report.md` 是给人看的最终研究报告。它展示 WriterAgent 的输出，并附带 JudgeAgent 的评分结果。

`demo_trace.json` 是给开发者和系统调试看的执行轨迹。它记录：

- 输入问题
- backend 类型
- Planner 生成的子任务
- Researcher 收集到的证据
- Judge 的最终评分
- 每一步执行状态
- 输出文件路径

在复杂多 Agent 系统中，trace 非常重要。它能帮助我们复盘系统为什么给出某个结果，也能帮助后续做调试、评测、失败恢复和可观测性建设。
