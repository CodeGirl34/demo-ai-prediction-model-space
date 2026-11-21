"""Sequence-level evaluation used in Hundman et al. (2018).

A labeled anomaly is a true positive if any predicted sequence overlaps it.
A predicted sequence that overlaps no labeled anomaly is a false positive.
"""


def _overlaps(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def evaluate(pred: list[tuple[int, int]], true: list[list[int]]) -> dict:
    tp = sum(any(_overlaps(t, p) for p in pred) for t in true)
    fp = sum(not any(_overlaps(p, t) for t in true) for p in pred)
    fn = len(true) - tp
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn}


def summarize(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f05 = 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "f0.5": f05}
