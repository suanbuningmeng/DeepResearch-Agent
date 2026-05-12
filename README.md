# DeepResearch-Agent

DeepResearch-Agent is a staged MVP for a multi-agent deep research workflow. It is designed to run a complete research loop without any real LLM API calls.

Current stage: **Stage 3.5 MVP with mock backend by default plus optional real OpenAI-compatible LLM backends**.

## What It Does

Given a research question, the demo runs this pipeline:

```text
User Question
-> Planner Agent creates 3-5 subtasks
-> TaskDAG stores research tasks and dependencies
-> DAGTaskScheduler runs Researcher tasks concurrently
-> optional ReplannerAgent creates replacement tasks after failure or evidence shortage
-> Memory Store saves evidence
-> Writer Agent generates a Markdown report
-> Judge Agent scores the report
-> outputs/demo_report.md and outputs/demo_trace.json
```

The default backend is `mock`. Real LLM backends are available only for manual demos and are not required for tests.

## Stage 2 Features

- DAG task graph for research subtasks.
- 9-state task model using `PENDING`, `READY`, `RUNNING`, `SUCCEEDED`, `FAILED`, `TIMEOUT`, `DEGRADED`, `REPLANNED`, and `CANCELLED`.
- Legal state transitions enforced by `TaskStateMachine`.
- `asyncio` research scheduling with `Semaphore` concurrency control.
- Per-task timeout handling.
- Retry support through `retry_count` and `max_retries`.
- Degraded fallback evidence when a task fails beyond retries or times out.
- Richer `demo_trace.json` with task state transitions and task timing.

## Stage 3 Features

- Dynamic Replan: failed, degraded, or evidence-insufficient runs can create replacement research tasks.
- Three-level degradation strategy:
  - Level 1: single task failure or timeout produces degraded fallback evidence.
  - Level 2: batch failure or evidence shortage triggers replanning.
  - Level 3: global timeout cancels unfinished tasks and forces partial report composition.
- Deterministic failure scenarios for local testing.
- Trace fields for replan events, degradation records, cancelled tasks, forced compose, and evidence sufficiency.

## Stage 3.5 Features

- OpenAI-compatible chat completions client built on `httpx.AsyncClient`.
- DeepSeek backend wrapper using OpenAI-compatible request format.
- Local vLLM backend wrapper using OpenAI-compatible request format.
- `create_llm()` factory for `mock`, `openai-compatible`, `deepseek`, and `vllm`.
- Safer JSON parsing for real LLM outputs that may include fenced JSON blocks.
- Existing self-built Agent orchestration is preserved. External Agent frameworks are not used.

## Install

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run Demo

By default, the demo runs in DAG mode:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock
```

You can also specify DAG mode and concurrency explicitly:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock --mode dag --max-concurrency 3
```

Failure scenario demos:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock --mode dag --failure-scenario fail_one
```

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock --mode dag --failure-scenario fail_many
```

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock --mode dag --failure-scenario timeout_one
```

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock --mode dag --failure-scenario global_timeout --global-timeout-seconds 1
```

To run the original first-stage linear flow:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock --mode linear
```

## Real LLM Backends

The `mock` backend is still the default and should be used for local tests. `deepseek`, `openai-compatible`, and `vllm` are intended for manual demos with your own configuration.

Configure environment variables from `.env.example`:

```bash
DEEPSEEK_API_KEY=
DEEPSEEK_API_BASE=
DEEPSEEK_MODEL=

OPENAI_COMPATIBLE_API_KEY=
OPENAI_COMPATIBLE_API_BASE=
OPENAI_COMPATIBLE_MODEL=

VLLM_API_BASE=http://localhost:8000/v1
VLLM_MODEL=Qwen2.5-7B-Instruct
```

Mock demo:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock --mode dag
```

DeepSeek demo:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend deepseek --mode dag --model deepseek-chat
```

Generic OpenAI-compatible demo:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend openai-compatible --mode dag --api-base "https://api.example.com/v1" --api-key-env "MY_API_KEY" --model "my-model"
```

Local vLLM demo:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend vllm --mode dag --api-base "http://localhost:8000/v1" --model "Qwen2.5-7B-Instruct"
```

SiliconFlow Qwen3 demo:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend openai-compatible --mode linear --api-base https://api.siliconflow.cn/v1 --api-key-env SILICONFLOW_API_KEY --model Qwen/Qwen3-8B --max-tokens 512 --request-timeout 180 --disable-thinking
```

For Qwen3 models, start with `--disable-thinking` to reduce long hidden-reasoning output and timeout risk. For the first real API test, run `linear` mode first. After linear mode works, try `dag` mode with `--max-concurrency 1` before increasing concurrency.

Recommended stable SiliconFlow Qwen3 DAG demo:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend openai-compatible --mode dag --max-concurrency 1 --api-base https://api.siliconflow.cn/v1 --api-key-env SILICONFLOW_API_KEY --model Qwen/Qwen3-8B --max-tokens 512 --request-timeout 180 --global-timeout-seconds 300 --writer-top-k-per-task 2 --disable-thinking
```

The real-LLM path includes extra stability handling: thinking-block stripping, fenced JSON cleanup, trailing-comma repair, fallback parsing for unstructured Researcher output, Writer evidence Top-K selection, and report completeness checks.

Important: Agent orchestration is self-built. This project does not use LangChain Agent, OpenAI Agents SDK, CrewAI, AutoGen, or similar external Agent frameworks.

### Linear Mode vs DAG Mode

`linear` mode runs each Researcher task one by one, preserving the first-stage MVP flow.

`dag` mode builds a `TaskDAG`, promotes dependency-ready tasks through the state machine, and executes ready Researcher tasks concurrently with bounded `asyncio` concurrency.

This creates:

- `outputs/demo_report.md`: Markdown research report with the judge score appended.
- `outputs/demo_trace.json`: execution trace containing the input question, planned subtasks, collected evidence, judge score, backend, execution mode, execution steps, and DAG task state details.

Key DAG trace fields include:

- `task_state_transitions`: every recorded task state change.
- `final_task_states`: final state for each original or replanned task.
- `degradation_records`: degraded fallback details.
- `replan_events`: dynamic replan events and generated replacement task IDs.
- `forced_compose`: whether global timeout forced partial composition.
- `cancelled_tasks`: tasks cancelled by global timeout or unreachable DAG state.
- `evidence_sufficiency`: evidence count and sufficiency status.

## Run Tests

```bash
pytest --basetemp=.pytest_tmp_llm
```

## Documentation

- [MVP code walkthrough (Chinese)](docs/mvp_code_walkthrough_zh.md)
- [Agent concepts (Chinese)](docs/agent_concepts_zh.md)
- [Interview notes (Chinese)](docs/interview_notes_zh.md)

## Project Layout

```text
DeepResearch-Agent/
|-- README.md
|-- pyproject.toml
|-- .env.example
|-- docs/
|-- scripts/
|   `-- run_demo.py
|-- src/
|   `-- deepresearch_agent/
|       |-- schemas.py
|       |-- engine/
|       |-- llm/
|       |-- agents/
|       |-- memory/
|       `-- report/
`-- tests/
```

## Roadmap

Future stages can add:

- Red-Blue adversarial research and critique loops.
- Shared memory with persistence and retrieval.
- SQLite or vector-store backed evidence storage.
- ResearchBench-style evaluation and regression tests.

These are intentionally not implemented in the first-stage MVP.
