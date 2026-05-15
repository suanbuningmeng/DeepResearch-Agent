from deepresearch_agent.evaluation.statistics import bootstrap_ci, cohens_d, mean_std


def test_bootstrap_ci_normal_case() -> None:
    low, high = bootstrap_ci([1, 2, 3, 4], n_bootstrap=100, seed=1)

    assert low <= high
    assert low >= 1
    assert high <= 4


def test_cohens_d_normal_case() -> None:
    value = cohens_d([1, 2, 3], [2, 3, 4])

    assert value > 0


def test_statistics_edge_cases() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([5]) == (5.0, 5.0)
    assert cohens_d([1], [2]) == 0.0
    assert mean_std([])["count"] == 0
