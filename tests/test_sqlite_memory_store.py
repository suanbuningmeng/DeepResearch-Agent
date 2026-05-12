from pathlib import Path

from deepresearch_agent.memory import SQLiteMemoryStore
from deepresearch_agent.schemas import Evidence


def make_evidence(evidence_id: str = "e1") -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id="task_1",
        title="Long-context benchmark",
        content="A benchmark should test retrieval and synthesis.",
        source_url="https://arxiv.org/abs/example",
        confidence=0.9,
        metadata={"kind": "test"},
    )


def test_sqlite_memory_store_round_trip(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(
        db_path=str(tmp_path / "memory.sqlite"),
        vector_index_path=str(tmp_path / "vector_index.npz"),
    )

    assert store.add_evidence(make_evidence()) is True
    evidences = store.list_evidences()
    loaded = store.get_evidence("e1")

    assert len(evidences) == 1
    assert loaded is not None
    assert loaded.metadata == {"kind": "test"}

    store.clear()
    assert store.list_evidences() == []
