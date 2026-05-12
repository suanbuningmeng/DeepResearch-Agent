# Agent 基础概念导读

这份文档用初学者视角解释 LLM Agent，并结合 DeepResearch-Agent 当前 MVP 说明每个概念在代码中如何体现。

## 1. 什么是 LLM Agent

LLM Agent 可以理解为“带有角色、目标和执行流程的 LLM 调用单元”。普通 LLM 调用通常只是输入一个 prompt，得到一个回答；Agent 则会被设计成某个明确角色，例如规划者、研究者、写作者或评审者。

在本项目中：

- `PlannerAgent` 负责拆解问题。
- `ResearcherAgent` 负责产生证据。
- `WriterAgent` 负责写报告。
- `JudgeAgent` 负责评分。

每个 Agent 都有清晰的输入、输出和职责边界。

## 2. Agent 和普通 LLM 调用有什么区别

普通 LLM 调用更像一次性问答：

```text
prompt -> LLM -> answer
```

Agent 更像一个可组合的功能模块：

```text
structured input -> role-specific prompt -> LLM -> structured output
```

区别主要有三点：

- Agent 有固定职责，不同 Agent 处理不同阶段。
- Agent 通常接收和返回结构化数据，而不只是自然语言。
- 多个 Agent 可以组成流程，形成更复杂的系统行为。

当前 MVP 虽然使用 MockLLM，但 Agent 的组织方式已经为后续接入真实模型打好了接口基础。

## 3. 什么是多 Agent 协作

多 Agent 协作是指多个 Agent 分工完成一个复杂任务。它不是让一个模型一次性回答所有内容，而是把任务拆成多个步骤：

```text
规划 -> 研究 -> 记忆 -> 写作 -> 评审
```

这种方式的好处是：

- 每一步职责更清楚。
- 中间产物可检查，例如 subtasks 和 evidence。
- 更容易做调试、重试和评测。
- 后续可以加入并发、对抗评审、工具调用和长期记忆。

## 4. Planner / Researcher / Writer / Judge 的角色

`PlannerAgent` 类似项目经理或研究主管。它不直接写最终报告，而是把大问题拆成可执行的子问题。

`ResearcherAgent` 类似研究助理。它针对每个子任务收集证据。当前证据是 mock 的，后续可以接搜索、论文数据库、网页抓取或内部知识库。

`WriterAgent` 类似报告作者。它把问题和证据组织成结构化 Markdown 报告。

`JudgeAgent` 类似审稿人或质量评估器。它检查报告质量，并给出多维度评分。

## 5. 为什么需要 trace

trace 是系统执行过程的记录。对于复杂 Agent 系统来说，只看最终答案是不够的，还需要知道答案是怎么来的。

当前 `demo_trace.json` 记录了：

- 用户输入了什么问题。
- Planner 拆出了哪些任务。
- Researcher 为每个任务生成了哪些证据。
- Writer 和 Judge 是否成功执行。
- Judge 给出了什么分数。

trace 的价值在于：

- 方便调试失败原因。
- 方便解释系统行为。
- 方便做自动化评测。
- 方便后续支持重试、恢复和可观测性。

## 6. 为什么需要 MemoryStore

MemoryStore 是 Agent 之间共享信息的地方。ResearcherAgent 生成的 evidence 不能只存在于局部变量里，否则 WriterAgent 和 JudgeAgent 无法复用。

当前 MemoryStore 很简单，只是内存 list。但它已经表达了一个重要概念：多 Agent 系统需要一个共享上下文或共享记忆层。

后续 MemoryStore 可以扩展为：

- JSON 文件存储
- SQLite 数据库
- 向量数据库
- 支持检索和去重的共享记忆系统

## 7. 当前 MVP 和完整 DeepResearch Agent 的差距

当前 MVP 已经跑通了最小流程，但还不是完整深度研究系统。主要差距包括：

- 证据是 mock 的，不来自真实资料。
- 子任务是线性执行，不支持 DAG 并发调度。
- 没有复杂的失败恢复、重试和降级逻辑。
- 没有 Red-Blue 对抗审查。
- 记忆层只是内存 list，不能长期保存和检索。
- 没有标准化 benchmark 或自动评测体系。

换句话说，当前版本证明的是“架构闭环可运行”，不是“研究质量已经可用于生产”。

## 8. 后续高级能力分别解决什么问题

DAG 并发调度解决的是复杂任务执行效率和依赖管理问题。不同子任务之间可能有依赖关系，也可能可以并发执行。DAG 能让系统知道哪些任务能同时跑，哪些必须等前置任务完成。

Red-Blue 对抗解决的是质量和鲁棒性问题。Blue 侧负责生成研究结论，Red 侧负责质疑、找漏洞、指出证据不足，从而提升报告可靠性。

共享记忆解决的是长期上下文和跨 Agent 协作问题。不同 Agent 可以读写同一个证据池、假设池、反驳记录和历史任务记录。

评测体系解决的是“如何知道系统变好了”的问题。ResearchBench 风格评测可以用固定任务集、评分标准和回归测试来衡量系统改动是否提升了质量。

这些能力都属于后续阶段，当前 MVP 只保留接口和结构上的扩展空间。
