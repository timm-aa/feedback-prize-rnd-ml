"""Span-level macro F1 with bidirectional overlap >= 0.5.

    Implementation from the competition's written definition, not a copied
    notebook. Matching is one-to-one, descending (max overlap, min overlap),
    with original indices as deterministic tie breakers.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
from .data import CLASSES


def _groups(frame, class_column):
    groups = defaultdict(list)
    for row in frame[["id", class_column, "predictionstring"]].itertuples(index=False, name=None):
        groups[(str(row[0]), row[1])].append(set(map(int, str(row[2]).split())))
    return groups


def score_spans(gold, predictions, essay_ids=None):
    g = _groups(gold, "discourse_type")
    p = _groups(predictions, "class")
    ids = sorted(set(essay_ids) if essay_ids is not None else
                 {key[0] for key in set(g) | set(p)})
    counts = np.zeros((len(ids), len(CLASSES), 3), dtype=np.int64)
    for d, essay_id in enumerate(ids):
        for c, label in enumerate(CLASSES):
            truth, pred = g[(essay_id, label)], p[(essay_id, label)]
            edges = []
            for i, a in enumerate(truth):
                for j, b in enumerate(pred):
                    inter = len(a & b)
                    if a and b and 2 * inter >= len(a) and 2 * inter >= len(b):
                        oa, ob = inter / len(a), inter / len(b)
                        edges.append((-max(oa, ob), -min(oa, ob), i, j))
            used_g, used_p = set(), set()
            for _, _, i, j in sorted(edges):
                if i not in used_g and j not in used_p:
                    used_g.add(i)
                    used_p.add(j)
            tp = len(used_g)
            counts[d, c] = tp, len(pred) - tp, len(truth) - tp
    return metrics_from_counts(counts.sum(axis=0)), counts, ids


def metrics_from_counts(counts):
    tp, fp, fn = counts.T
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(CLASSES)), where=(2 * tp + fp + fn) > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros(len(CLASSES)), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(len(CLASSES)), where=(tp + fn) > 0)
    per_class = {label: {"f1": float(f1[i]), "precision": float(precision[i]),
                         "recall": float(recall[i]), "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i])}
                 for i, label in enumerate(CLASSES)}
    denom = int((2 * tp + fp + fn).sum())
    return {"macro_f1": float(f1.mean()), "micro_f1": float(2 * tp.sum() / denom) if denom else 0.,
            "per_class": per_class}


def bootstrap_scores(counts_by_model, groups, iterations, seed):
    """Paired cluster bootstrap; normalized duplicate essays form one unit."""
    unique, inverse = np.unique(groups, return_inverse=True)
    clustered = {}
    for name, counts in counts_by_model.items():
        aggregate = np.zeros((len(unique), len(CLASSES), 3), dtype=np.int64)
        np.add.at(aggregate, inverse, counts)
        clustered[name] = aggregate
    rng = np.random.default_rng(seed)
    values = {name: [] for name in clustered}
    for _ in range(iterations):
        sampled = rng.integers(0, len(unique), len(unique))
        for name, counts in clustered.items():
            values[name].append(metrics_from_counts(counts[sampled].sum(axis=0))["macro_f1"])
    return {name: np.array(scores) for name, scores in values.items()}
