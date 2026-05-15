from __future__ import annotations

import math
import random


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return (0.0, 0.0)
    if len(clean) == 1:
        return (clean[0], clean[0])
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, n_bootstrap)):
        sample = [rng.choice(clean) for _ in clean]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = 1.0 - confidence
    low_index = int((alpha / 2) * (len(means) - 1))
    high_index = int((1 - alpha / 2) * (len(means) - 1))
    return (round(means[low_index], 6), round(means[high_index], 6))


def cohens_d(
    baseline: list[float],
    treatment: list[float],
) -> float:
    left = [float(value) for value in baseline if value is not None]
    right = [float(value) for value in treatment if value is not None]
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left_std = _sample_std(left)
    right_std = _sample_std(right)
    pooled = math.sqrt(((len(left) - 1) * left_std**2 + (len(right) - 1) * right_std**2) / (len(left) + len(right) - 2))
    if pooled == 0.0:
        return 0.0
    return round((_mean(right) - _mean(left)) / pooled, 6)


def mean_std(values: list[float]) -> dict:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": round(_mean(clean), 6),
        "std": round(_sample_std(clean), 6) if len(clean) > 1 else 0.0,
        "count": len(clean),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
