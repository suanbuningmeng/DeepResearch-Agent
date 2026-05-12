from pathlib import Path

from deepresearch_agent.memory import HashingEmbeddingProvider, NumpyVectorIndex


def test_hashing_embedding_provider_is_stable() -> None:
    provider = HashingEmbeddingProvider(dim=32)

    first = provider.embed_text("long context evaluation")
    second = provider.embed_text("long context evaluation")

    assert first.shape == (32,)
    assert (first == second).all()


def test_numpy_vector_index_add_search_save_load(tmp_path: Path) -> None:
    provider = HashingEmbeddingProvider(dim=32)
    index = NumpyVectorIndex()
    index.add("e1", provider.embed_text("retrieval benchmark"))
    index.add("e2", provider.embed_text("cooking recipe"))

    results = index.search(provider.embed_text("retrieval benchmark"), top_k=1)
    assert results[0][0] == "e1"

    path = tmp_path / "index.npz"
    index.save(str(path))
    loaded = NumpyVectorIndex.load(str(path))
    loaded_results = loaded.search(provider.embed_text("retrieval benchmark"), top_k=1)

    assert loaded_results[0][0] == "e1"
