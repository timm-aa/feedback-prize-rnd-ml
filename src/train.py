"""Run fixed 3-fold comparisons and retrain the selected model on all train."""
import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .data import CLASSES, LABELS, decode, load_training
from .metrics import bootstrap_scores, score_spans
from .model import Baseline


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def make_folds(docs, config):
    splitter = StratifiedGroupKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
    docs = docs.copy()
    docs["fold"] = -1
    for fold, (tr, va) in enumerate(splitter.split(docs, docs.stratum, docs.group)):
        assert not set(docs.iloc[tr].group) & set(docs.iloc[va].group)
        docs.loc[va, "fold"] = fold
    assert (docs.fold >= 0).all()
    assert docs.groupby("group").fold.nunique().max() == 1
    return docs


def run(config):
    start = time.perf_counter()
    root, artifact = Path(config["output_dir"]), Path(config["artifacts_dir"])
    for folder in [root, root / "splits", root / "private", root / "predictions", artifact]:
        folder.mkdir(exist_ok=True, parents=True)
    print("Loading and auditing every training essay...", flush=True)
    gold, frame, docs, lengths, audit = load_training(config["data_dir"])
    docs = make_folds(docs, config)
    fold_map = docs.set_index("id").fold
    frame["fold"] = frame.id.map(fold_map)
    docs.to_csv(root / "splits" / "essay_folds.csv", index=False)
    lengths.groupby("class").words.describe().to_csv(root / "span_lengths.csv")
    save_json(root / "data_audit.json", audit)
    save_json(root / "config_resolved.json", config)
    print(f"Audited {len(docs)} essays, {len(gold)} spans, {len(frame)} candidates", flush=True)
    print(f"Audit: {audit}", flush=True)

    all_results, oof_labels, fold_rows = {}, {}, []
    counts_by_model = {}
    sorted_ids = sorted(docs.id)
    for name in config["experiments"]:
        labels = np.full(len(frame), "O", dtype=object)
        elapsed = 0.
        for fold in range(config["n_splits"]):
            t = time.perf_counter()
            train = frame.loc[frame.fold != fold]
            val = frame.loc[frame.fold == fold]
            val_ids = set(docs.loc[docs.fold == fold, "id"])
            assert not set(train.id) & val_ids
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                with threadpool_limits(limits=2):
                    model = Baseline(name, config).fit(train)
                    pred = model.predict(val)
            labels[val.index] = pred
            spans = decode(val, pred)
            metrics, _, _ = score_spans(gold.loc[gold.id.isin(val_ids)], spans, val_ids)
            duration = time.perf_counter() - t
            elapsed += duration
            row = {"experiment": name, "fold": fold, "train_essays": len(set(train.id)),
                   "val_essays": len(val_ids), "span_macro_f1": metrics["macro_f1"],
                   "span_micro_f1": metrics["micro_f1"],
                   "candidate_macro_f1": float(f1_score(val.target, pred, labels=LABELS, average="macro", zero_division=0)),
                   "seconds": duration, "warnings": [str(w.message) for w in caught],
                   "epochs": int(getattr(model.classifier, "n_iter_", 0))}
            fold_rows.append(row)
            pd.DataFrame(fold_rows).to_csv(root / "fold_metrics.csv", index=False)
            print(f'{name} fold={fold}: span macro F1={metrics["macro_f1"]:.5f}, {duration:.1f}s', flush=True)
        spans = decode(frame, labels)
        spans.to_csv(root / "predictions" / f"{name}_oof.csv", index=False)
        metrics, counts, ids = score_spans(gold, spans, sorted_ids)
        assert ids == sorted_ids
        counts_by_model[name] = counts
        np.save(artifact / f"{name}_oof_counts.npy", counts)
        oof_labels[name] = labels
        fold_f1 = [r["span_macro_f1"] for r in fold_rows if r["experiment"] == name]
        metrics.update({"fold_mean": float(np.mean(fold_f1)), "fold_std": float(np.std(fold_f1, ddof=1)),
                        "fold_values": fold_f1, "total_seconds": elapsed,
                        "candidate_macro_f1": float(f1_score(frame.target, labels, labels=LABELS, average="macro", zero_division=0)),
                        "candidate_accuracy": float(accuracy_score(frame.target, labels))})
        all_results[name] = metrics
        save_json(root / "metrics.json", all_results)
    best = max(all_results, key=lambda name: all_results[name]["fold_mean"])
    print(f"Selected by mean validation span F1: {best}", flush=True)

    groups = docs.set_index("id").loc[sorted_ids, "group"].to_numpy()
    boots = bootstrap_scores(counts_by_model, groups, config["bootstrap_iterations"], config["seed"])
    for name in all_results:
        all_results[name]["bootstrap_ci95"] = np.quantile(boots[name], [.025, .975]).tolist()
    pairs = [("lexical", "dummy"), ("lexical_position", "lexical"),
             ("lexical_position_balanced", "lexical_position")]
    deltas = {}
    for a, b in pairs:
        if a in boots and b in boots:
            deltas[f"{a}_minus_{b}"] = {
                "delta_oof": all_results[a]["macro_f1"] - all_results[b]["macro_f1"],
                "paired_bootstrap_ci95": np.quantile(boots[a] - boots[b], [.025, .975]).tolist()}
    save_json(root / "metrics.json", all_results)
    save_json(root / "paired_comparisons.json", deltas)

    diagnostic, _, _ = score_spans(gold, decode(frame, frame.target), sorted_ids)
    save_json(root / "candidate_gold_diagnostic.json", diagnostic)
    selected_spans = pd.read_csv(root / "predictions" / f"{best}_oof.csv")
    slices = {}
    for low, high, title in [(0, 250, "short_lt250"), (250, 500, "medium_250_499"), (500, 10**9, "long_ge500")]:
        ids = set(docs.loc[(docs.words >= low) & (docs.words < high), "id"])
        metrics, _, _ = score_spans(gold.loc[gold.id.isin(ids)], selected_spans.loc[selected_spans.id.isin(ids)], ids)
        slices[title] = {"documents": len(ids), **metrics}
    save_json(root / "length_slices.json", slices)
    matrix = confusion_matrix(frame.target, oof_labels[best], labels=LABELS)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(root / "candidate_confusion.csv")
    private = frame.copy()
    private["prediction"] = oof_labels[best]
    private.to_pickle(root / "private" / "candidates_oof.pkl")
    docs.to_pickle(artifact / "documents.pkl")
    lengths.to_pickle(artifact / "span_lengths.pkl")

    print("Retraining selected configuration on all training essays...", flush=True)
    with threadpool_limits(limits=2):
        model = Baseline(best, config).fit(frame)
    joblib.dump(model, artifact / "baseline.joblib", compress=3)
    selected = {"experiment": best, "criterion": "mean 3-fold validation span macro F1",
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    save_json(root / "selected_model.json", selected)
    import scipy, matplotlib, threadpoolctl
    save_json(root / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "elapsed_seconds": time.perf_counter() - start,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("src").glob("*.py"))}})
    print(json.dumps({"best": best, "results": {k: v["fold_mean"] for k, v in all_results.items()},
                      "seconds": time.perf_counter() - start}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.json")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
