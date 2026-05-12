# DeepResearch-Agent 面试讲法

这份文档帮助你在面试中清楚介绍当前项目。重点是讲明白：为什么要做多 Agent、第一阶段实现了什么、为什么先用 mock backend、后续如何扩展。

## 1. 30 秒项目介绍

DeepResearch-Agent 是一个面向复杂深度研究任务的多智能体协作系统。我目前实现的是第一阶段 MVP：用户输入一个研究问题后，系统会由 Planner 拆分子任务，Researcher 生成 mock evidence，MemoryStore 保存证据，Writer 生成 Markdown 报告，Judge 对报告打分，最后输出报告和执行 trace。当前版本默认使用 MockLLM，不依赖真实 API，重点是先跑通端到端架构闭环。

## 2. 1 分钟项目介绍

DeepResearch-Agent 的目标是把复杂研究任务拆成多个可观察、可评估、可扩展的 Agent 步骤，而不是让一个 LLM 一次性回答所有内容。当前 MVP 包含四个 Agent：Planner 负责把问题拆成 3-5 个子任务；Researcher 为每个子任务生成结构化证据；MemoryStore 负责保存证据；Writer 根据证据生成 Markdown 研究报告；Judge 从事实性、覆盖度、推理深度、引用质量和清晰度等维度给报告打分。

第一阶段我刻意使用 MockLLM，避免 API Key、网络和模型不稳定性影响架构验证。运行 demo 后会生成 `demo_report.md` 和 `demo_trace.json`。前者给用户阅读，后者记录系统的执行过程，方便调试和后续评测。后续计划是在这个稳定闭环上扩展 DAG 并发调度、Red-Blue 对抗、共享记忆和 ResearchBench 风格评测。

## 3. 当前 MVP 已经实现了什么

当前 MVP 已经实现：

- Pydantic schema，包括任务、证据、评分和报告。
- 统一 LLM 抽象接口 `BaseLLM`。
- 默认 mock backend `MockLLM`。
- 四个 Agent：Planner、Researcher、Writer、Judge。
- 简单内存证据存储 `MemoryStore`。
- 命令行 demo：`scripts/run_demo.py`。
- 两个输出文件：`demo_report.md` 和 `demo_trace.json`。
- pytest 测试，覆盖 MockLLM、Agent 串联和 demo 输出。

## 4. 为什么第一阶段先使用 MockLLM

我第一阶段使用 MockLLM 是为了先验证系统架构，而不是过早陷入真实模型调用的复杂性。真实 API 会引入 API Key、网络、费用、速率限制和输出不稳定等变量，这些会干扰 MVP 阶段最核心的问题：数据结构是否合理、Agent 边界是否清晰、流程是否能端到端跑通、trace 是否能记录关键过程。

MockLLM 让系统具备确定性，测试也更稳定。等架构闭环稳定后，再接 OpenAI、DeepSeek 或 vLLM 会更自然。

## 5. 这个项目和普通 ChatGPT 调用有什么区别

普通 ChatGPT 调用通常是一次 prompt 得到一次回答。这个项目把研究过程拆成多个角色和中间产物：

- Planner 输出结构化子任务。
- Researcher 输出结构化 evidence。
- MemoryStore 保存中间证据。
- Writer 使用证据生成报告。
- Judge 独立评分。
- trace 记录完整执行链路。

因此，它不是单次问答，而是一个可组合、可观测、可测试的多 Agent 工作流。

## 6. 当前项目的不足

当前不足主要有：

- 证据是 mock 的，还没有真实检索能力。
- Planner 只生成简单子任务，没有复杂依赖关系。
- 子任务是顺序执行，没有 DAG 并发调度。
- MemoryStore 是内存 list，不支持持久化和检索。
- Judge 评分也是 mock 的，不能代表真实质量评估。
- 没有 Red-Blue 对抗机制，也没有正式 benchmark。

这些不足是第一阶段刻意保留的边界，因为当前目标是跑通最小闭环。

## 7. 后续计划怎么扩展

DAG 并发调度：把 Planner 输出从简单列表升级为带依赖关系的任务图。调度器可以根据依赖关系并发执行独立任务，并处理失败、超时和重试。

Red-Blue 对抗：加入生成方和审查方。Blue Agent 负责提出结论和写报告，Red Agent 负责攻击论证漏洞、证据不足和潜在幻觉，最后由 Writer 或 Judge 汇总修正。

ResearchBench：设计一组固定研究任务和评分标准，用来评估系统改动前后的质量变化。它可以帮助避免“感觉变好了但没有证据”的问题。

共享记忆：把当前 MemoryStore 升级为持久化、可检索、可去重的证据和上下文系统，让多个 Agent 可以共享资料、历史结论和反驳记录。

## 8. 面试官可能追问的问题和回答

**问：为什么不直接用一个 prompt 让 LLM 写报告？**

答：单 prompt 很难观察中间过程，也不容易复用、调试和评测。多 Agent 工作流把规划、证据、写作和评审拆开，可以检查每一步的质量，也方便后续做并发、重试和对抗评审。

**问：当前使用 mock evidence，有实际价值吗？**

答：有。第一阶段的目标不是证明研究内容质量，而是证明架构闭环、数据结构和测试体系可行。MockLLM 提供确定性输出，让我可以稳定验证流程。真实 evidence 会在下一阶段接入。

**问：为什么需要 JudgeAgent？**

答：JudgeAgent 让系统具备初步自评估能力。即使当前评分是 mock，它也确定了评估接口和评分维度。未来可以替换成真实模型评审、规则评审或 benchmark 评分。

**问：MemoryStore 现在只是 list，会不会太简单？**

答：对第一阶段来说足够。它表达了“Agent 之间需要共享证据”的核心抽象。后续可以在不改变 Agent 主流程的情况下，把底层替换成 SQLite、向量库或持久化记忆。

**问：如何保证后续接真实 LLM 不破坏系统？**

答：关键是 `BaseLLM` 抽象。Agent 只依赖 `agenerate` 接口，不直接依赖某个厂商 SDK。后续新增 OpenAI、DeepSeek 或 vLLM client 时，只要实现同样接口，并保持返回格式，就能平滑替换。

**问：你会如何处理真实模型输出不稳定？**

答：我会加入结构化输出校验、JSON 修复、重试、超时、降级策略和 trace 记录。Pydantic schema 已经为结构化校验打了基础。

**问：DAG 调度为什么重要？**

答：复杂研究任务通常不是单条线。有些任务可以并行，有些任务依赖前置结果。DAG 调度能提升执行效率，也能更清楚地表达任务依赖和失败传播。

**问：Red-Blue 对抗会带来什么提升？**

答：它可以减少单一路径生成带来的偏差。Red Agent 专门质疑证据、寻找反例和识别逻辑漏洞，Blue Agent 再修正结论，这会提高报告的可靠性。

**问：ResearchBench 怎么设计？**

答：可以先设计固定问题集、参考证据、评分 rubrics 和自动评分脚本。每次改动系统后跑同一批任务，比较覆盖度、事实性、引用质量和稳定性等指标。
