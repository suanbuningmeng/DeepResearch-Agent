# DeepResearch-Agent

DeepResearch-Agent is a first-stage MVP for a multi-agent deep research workflow. It is designed to run a complete minimum loop without any real LLM API calls.

Current stage: **MVP with mock backend only**.

## What It Does

Given a research question, the demo runs this pipeline:

```text
User Question
-> Planner Agent creates 3-5 subtasks
-> Researcher Agent creates mock evidence for each subtask
-> Memory Store saves evidence
-> Writer Agent generates a Markdown report
-> Judge Agent scores the report
-> outputs/demo_report.md and outputs/demo_trace.json
```

The default and only implemented backend is `mock`.

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

```bash
python scripts/run_demo.py --question "What are the main challenges and recent methods for long-context LLM evaluation?" --backend mock
```

This creates:

- `outputs/demo_report.md`: Markdown research report with the judge score appended.
- `outputs/demo_trace.json`: execution trace containing the input question, planned subtasks, collected evidence, judge score, backend, and execution steps.

## Run Tests

```bash
pytest
```

## Project Layout

```text
DeepResearch-Agent/
├── README.md
├── pyproject.toml
├── .env.example
├── scripts/
│   └── run_demo.py
├── src/
│   └── deepresearch_agent/
│       ├── schemas.py
│       ├── llm/
│       ├── agents/
│       ├── memory/
│       └── report/
└── tests/
```

## Roadmap

Future stages can add:

- DAG-based concurrent task scheduling.
- Red-Blue adversarial research and critique loops.
- Shared memory with persistence and retrieval.
- Real backends for OpenAI, DeepSeek, and vLLM.
- SQLite or vector-store backed evidence storage.
- ResearchBench-style evaluation and regression tests.

These are intentionally not implemented in the first-stage MVP.
