"""Token-sequence metrics. These are not official CROHME symbol-graph metrics."""
import statistics
from .data import tokens


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


def score_example(truth, prediction):
    target, predicted = tokens(truth), tokens(prediction)
    distance = edit_distance(target, predicted)
    return {"ed": distance, "target_tokens": len(target), "predicted_tokens": len(predicted),
            "ned": distance / max(1, len(target), len(predicted)),
            "fraction": "\\frac" in target, "superscript": "^" in target,
            "radical": "\\sqrt" in target, "long": len(target) >= 50}


def summarize(examples):
    if not examples:
        raise ValueError("No examples to evaluate.")
    n = len(examples)
    return {"n": n, "exact": sum(e["ed"] == 0 for e in examples) / n,
            "within_1_token": sum(e["ed"] <= 1 for e in examples) / n,
            "within_2_tokens": sum(e["ed"] <= 2 for e in examples) / n,
            "mean_ed": statistics.mean(e["ed"] for e in examples),
            "mean_ned": statistics.mean(e["ned"] for e in examples),
            "token_error_rate": sum(e["ed"] for e in examples) /
                                max(1, sum(e["target_tokens"] for e in examples))}
