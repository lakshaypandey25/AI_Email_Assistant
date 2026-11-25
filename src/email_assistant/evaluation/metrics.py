"""Pure metric functions (no dependencies) - reused by the benchmark and the live statistics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

Pair = tuple[str, str]  # (actual, predicted)


def rate(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator`` or ``None`` when undefined."""
    return None if not denominator else numerator / denominator


def accuracy(pairs: Sequence[Pair]) -> float | None:
    return rate(sum(1 for actual, predicted in pairs if actual == predicted), len(pairs))


def confusion_matrix(pairs: Iterable[Pair], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    """``matrix[actual][predicted] = count`` over ``labels``."""
    matrix = {a: {p: 0 for p in labels} for a in labels}
    for actual, predicted in pairs:
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1
    return matrix


def per_class(pairs: Sequence[Pair], labels: Sequence[str]) -> dict[str, dict[str, float | int | None]]:
    """Precision / recall / F1 / support for each label."""
    out: dict[str, dict[str, float | int | None]] = {}
    for label in labels:
        tp = sum(1 for a, p in pairs if a == label and p == label)
        fp = sum(1 for a, p in pairs if a != label and p == label)
        fn = sum(1 for a, p in pairs if a == label and p != label)
        precision, recall = rate(tp, tp + fp), rate(tp, tp + fn)
        f1 = None if precision is None or recall is None or (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def macro_f1(stats: dict[str, dict[str, float | int | None]]) -> float | None:
    """Mean F1 over classes that actually occur (support > 0); a class never predicted correctly counts as 0."""
    scores = [(s["f1"] or 0.0) for s in stats.values() if (s["support"] or 0) > 0]
    return sum(scores) / len(scores) if scores else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def median(values: Sequence[float]) -> float | None:
    return percentile(values, 50)