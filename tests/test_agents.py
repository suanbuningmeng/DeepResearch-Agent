import asyncio

from deepresearch_agent.agents import JudgeAgent, PlannerAgent, ResearcherAgent, WriterAgent
from deepresearch_agent.llm import MockLLM
from deepresearch_agent.memory import MemoryStore


def test_agents_can_run_minimum_workflow() -> None:
    async def run() -> None:
        question = "What are the main challenges and recent methods for long-context LLM evaluation?"
        llm = MockLLM()
        memory = MemoryStore()

        subtasks = await PlannerAgent(llm).plan(question)
        assert 3 <= len(subtasks) <= 5

        researcher = ResearcherAgent(llm)
        for task in subtasks:
            memory.add_evidences(await researcher.research(task))

        evidences = memory.list_evidences()
        assert len(evidences) >= len(subtasks)

        report = await WriterAgent(llm).write(question, evidences)
        assert "## Key Findings" in report

        score = await JudgeAgent(llm).judge(question, report, evidences)
        assert score.overall > 0
        assert "mock evidence" in score.comments

    asyncio.run(run())
