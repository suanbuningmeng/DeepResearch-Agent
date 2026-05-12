# DeepResearch-Agent

DeepResearch-Agent is a staged MVP for a multi-agent deep research workflow. It is designed to run a complete research loop without any real LLM API calls.

Current stage: **Stage 2 MVP with mock backend, DAG scheduling, and task state management**.

## What It Does

Given a research question, the demo runs this pipeline:

```text
User Question
-> Planner Agent creates 3-5 subtasks
-> TaskDAG stores research tasks and dependencies
-> DAGTaskScheduler runs Researcher tasks concurrently
-> Memory Store saves evidence
-> Writer Agent generates a Markdown report
-> Judge Agent scores the report
-> outputs/demo_report.md and outputs/demo_trace.json
```

The default and only implemented backend is `mock`. No real API calls are made.

## Stage 2 Features

- DAG task graph for research subtasks.
- 9-state task model using `PENDING`, `READY`, `RUNNING`, `SUCCEEDED`, `FAILED`, `TIMEOUT`, `DEGRADED`, `REPLANNED`, and `CANCELLED`.
- Legal state transitions enforced by `TaskStateMachine`.
- `asyncio` research scheduling with `Semaphore` concurrency control.
- Per-task timeout handling.
- Retry support through `retry_count` and `max_retries`.
- Degraded fallback evidence when a task fails beyond retries or times out.
- Richer `demo_trace.json` with task state transitions and task timing.

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

To run the original first-stage linear flow:

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock --mode linear
```

### Linear Mode vs DAG Mode

`linear` mode runs each Researcher task one by one, preserving the first-stage MVP flow.

`dag` mode builds a `TaskDAG`, promotes dependency-ready tasks through the state machine, and executes ready Researcher tasks concurrently with bounded `asyncio` concurrency.

This creates:

- `outputs/demo_report.md`: Markdown research report with the judge score appended.
- `outputs/demo_trace.json`: execution trace containing the input question, planned subtasks, collected evidence, judge score, backend, execution mode, execution steps, and DAG task state details.

## Run Tests

```bash
pytest --basetemp=.pytest_tmp
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
- Real backends for OpenAI, DeepSeek, and vLLM.
- SQLite or vector-store backed evidence storage.
- ResearchBench-style evaluation and regression tests.

These are intentionally not implemented in the first-stage MVP.
