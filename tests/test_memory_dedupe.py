from pathlib import Path

from deepresearch_agent.memory import SQLiteMemoryStore
from deepresearch_agent.memory.dedupe import content_hash, normalize_text
from deepresearch_agent.schemas import Evidence


def evidence(evidence_id: str) -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id="task_1",
        title="Same title",
        content="Same content",
        source_url="mock://source",
        confidence=0.8,
        metadata={},
    )


def test_content_hash_detects_same_title_and_content() -> None:
    assert normalize_text("  SAME   Text ") == "same text"
    assert content_hash("Title", "Content") == content_hash(" title ", "content")


def test_sqlite_memory_store_skips_duplicate_evidence(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(
        db_path=str(tmp_path / "memory.sqlite"),
        vector_index_path=str(tmp_path / "vector_index.npz"),
    )

    assert store.add_evidence(evidence("e1")) is True
    assert store.add_evidence(evidence("e2")) is False

    assert len(store.list_evidences()) == 1
    assert store.duplicate_evidence_count == 1
