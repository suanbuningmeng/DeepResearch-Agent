from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from deepresearch_agent.memory.base import BaseMemoryStore
from deepresearch_agent.memory.dedupe import content_hash
from deepresearch_agent.memory.embedding import BaseEmbeddingProvider, HashingEmbeddingProvider
from deepresearch_agent.memory.source_quality import SourceQuality, classify_source_url
from deepresearch_agent.memory.vector_index import NumpyVectorIndex
from deepresearch_agent.schemas import Evidence


class SQLiteMemoryStore(BaseMemoryStore):
    """SQLite-backed persistent evidence store with local vector retrieval."""

    def __init__(
        self,
        db_path: str = "data/memory.sqlite",
        vector_index_path: str = "data/vector_index.npz",
        embedding_provider: BaseEmbeddingProvider | None = None,
    ) -> None:
        self.db_path = db_path
        self.vector_index_path = vector_index_path
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self.inserted_evidence_count = 0
        self.duplicate_evidence_count = 0
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(vector_index_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.vector_index = self._build_vector_index()

    def add_evidence(self, evidence: Evidence) -> bool:
        digest = content_hash(evidence.title, evidence.content)
        if self._content_hash_exists(digest) or self.get_evidence(evidence.id) is not None:
            self.duplicate_evidence_count += 1
            return False

        source_quality = classify_source_url(evidence.source_url).value
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO evidences (
                    id, task_id, title, content, source_url, confidence,
                    metadata_json, source_quality, content_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.id,
                    evidence.task_id,
                    evidence.title,
                    evidence.content,
                    evidence.source_url,
                    evidence.confidence,
                    json.dumps(evidence.metadata, ensure_ascii=False),
                    source_quality,
                    digest,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        vector = self.embedding_provider.embed_text(f"{evidence.title}\n{evidence.content}")
        self.vector_index.add(evidence.id, vector)
        self.vector_index.save(self.vector_index_path)
        self.inserted_evidence_count += 1
        return True

    def add_evidences(self, evidences: list[Evidence]) -> dict:
        inserted_count = 0
        duplicate_count = 0
        for evidence in evidences:
            if self.add_evidence(evidence):
                inserted_count += 1
            else:
                duplicate_count += 1
        return {"inserted_count": inserted_count, "duplicate_count": duplicate_count}

    def list_evidences(self) -> list[Evidence]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, title, content, source_url, confidence, metadata_json
                FROM evidences
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
        return [self._row_to_evidence(row) for row in rows]

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, title, content, source_url, confidence, metadata_json
                FROM evidences
                WHERE id = ?
                """,
                (evidence_id,),
            ).fetchone()
        return self._row_to_evidence(row) if row else None

    def search_evidences(self, query: str, top_k: int = 5) -> list[Evidence]:
        query_vector = self.embedding_provider.embed_text(query)
        results = self.vector_index.search(query_vector, top_k=top_k)
        evidences = [self.get_evidence(evidence_id) for evidence_id, _score in results]
        return [evidence for evidence in evidences if evidence is not None]

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM evidences")
        self.vector_index = NumpyVectorIndex()
        self.vector_index.save(self.vector_index_path)
        self.inserted_evidence_count = 0
        self.duplicate_evidence_count = 0

    def source_quality_summary(self) -> dict[str, int]:
        summary = {quality.value: 0 for quality in SourceQuality}
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT source_quality, COUNT(*) FROM evidences GROUP BY source_quality"
            ).fetchall()
        for quality, count in rows:
            summary[str(quality)] = int(count)
        return summary

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidences (
                    id TEXT PRIMARY KEY,
                    task_id TEXT,
                    title TEXT,
                    content TEXT,
                    source_url TEXT,
                    confidence REAL,
                    metadata_json TEXT,
                    source_quality TEXT,
                    content_hash TEXT UNIQUE,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidences_content_hash ON evidences(content_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_evidences_source_quality ON evidences(source_quality)"
            )

    def _content_hash_exists(self, digest: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM evidences WHERE content_hash = ? LIMIT 1",
                (digest,),
            ).fetchone()
        return row is not None

    def _build_vector_index(self) -> NumpyVectorIndex:
        index = NumpyVectorIndex()
        for evidence in self.list_evidences():
            vector = self.embedding_provider.embed_text(f"{evidence.title}\n{evidence.content}")
            index.add(evidence.id, vector)
        index.save(self.vector_index_path)
        return index

    def _row_to_evidence(self, row: tuple) -> Evidence:
        metadata = json.loads(row[6]) if row[6] else {}
        return Evidence(
            id=row[0],
            task_id=row[1],
            title=row[2],
            content=row[3],
            source_url=row[4],
            confidence=float(row[5]),
            metadata=metadata,
        )
