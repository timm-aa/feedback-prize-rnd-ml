"""Fixed factorial experiment: context × selective decoder; no HPO/test tuning."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import f1_score, confusion_matrix
from threadpoolctl import threadpool_limits

from .data import LABELS, load_training, decode
from .metrics import score_spans, bootstrap_scores
from .model import Baseline
from .hw2_model import ContextBaseline, HW2Predictor, decode_selective
from .train import make_folds, save_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cfg):
    started = time.perf_counter()
    root, artifact = Path(cfg["output_dir"]), Path(cfg["artifacts_dir"])
    for folder in [root, root / "predictions", root / "private", root / "splits", artifact]:
        folder.mkdir(parents=True, exist_ok=True)
    print("Auditing raw training data and recreating the fixed HW1 folds...", flush=True)
    gold, frame, docs, _, audit = load_training(cfg["data_dir"])
    docs = make_folds(docs, cfg)
    prior = Path("reports/splits/essay_folds.csv")
    split_match = None
    if prior.exists():
        previous = pd.read_csv(prior).set_index("id").sort_index()
        current = docs.set_index("id").sort_index()
        pd.testing.assert_frame_equal(current[["group", "fold"]], previous[["group", "fold"]])
        split_match = True
    frame["fold"] = frame.id.map(docs.set_index("id").fold)
    split_path = root / "splits" / "essay_folds.csv"
    docs.to_csv(split_path, index=False)
    save_json(root / "config_resolved.json", cfg)
    save_json(root / "data_audit.json", audit)
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "seed": cfg["seed"],
                "split_sha256": sha(split_path), "same_split_as_hw1": split_match,
                "train_csv_sha256": audit["train_csv_sha256"], "text_manifest_sha256": audit["text_manifest_sha256"],
                "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                             "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
                "source_sha256": {str(p): sha(p) for p in sorted(Path("src").glob("*.py"))},
                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                "plan_sha256": sha("docs/HW2_PLAN.md"), "primary_metric": "span macro F1",
                "fit_policy": "train folds only; two fits per fold reused for decoder comparisons"}
    save_json(root / "run_metadata.json", metadata)
    (root / "events.jsonl").write_text("")
    oof = {False: np.full(len(frame), "O", dtype=object), True: np.full(len(frame), "O", dtype=object)}
    rows = []
    for fold in range(cfg["n_splits"]):
        train, val = frame.loc[frame.fold != fold], frame.loc[frame.fold == fold]
        val_ids = set(val.id)
        train_ids = set(train.id)
        assert not train_ids & val_ids
        assert not set(docs.loc[docs.id.isin(train_ids), "group"]) & set(docs.loc[docs.id.isin(val_ids), "group"])
        for context in [False, True]:
            fit_id = f'fold{fold}_{"context" if context else "baseline"}_seed{cfg["seed"]}'
            t = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with threadpool_limits(limits=2):
                    model = ContextBaseline(cfg) if context else Baseline("lexical_position_balanced", cfg)
                    model.fit(train)
                    fit_seconds = time.perf_counter() - t
                    pred = model.predict(val)
            predict_seconds = time.perf_counter() - t - fit_seconds
            oof[context][val.index] = pred
            n_features = model.classifier.coef_.shape[1]
            for experiment in [e for e in cfg["experiments"] if e["context"] == context]:
                evaluation_start = time.perf_counter()
                spans = (decode_selective(val, pred, cfg["selective_merge_classes"])
                         if experiment["selective_merge"] else decode(val, pred))
                metrics, _, _ = score_spans(gold.loc[gold.id.isin(val_ids)], spans, val_ids)
                row = {"experiment": experiment["name"], "hypothesis": experiment["hypothesis"],
                       "context": context, "selective_merge": experiment["selective_merge"],
                       "fold": fold, "seed": cfg["seed"], "fit_id": fit_id,
                       "train_essays": len(train_ids), "val_essays": len(val_ids),
                       "train_candidates": len(train), "val_candidates": len(val),
                       "vocabulary_size": len(model.tfidf.vocabulary_), "feature_count": n_features,
                       "epochs": int(model.classifier.n_iter_), "span_macro_f1": metrics["macro_f1"],
                       "span_micro_f1": metrics["micro_f1"],
                       "candidate_macro_f1": float(f1_score(val.target, pred, labels=LABELS, average="macro", zero_division=0)),
                       "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                       "decode_score_seconds": time.perf_counter() - evaluation_start,
                       "warnings": [str(w.message) for w in caught], "split_sha256": metadata["split_sha256"],
                       "config_sha256": metadata["config_sha256"],
                       "logged_utc": datetime.now(timezone.utc).isoformat()}
                for c, m in metrics["per_class"].items():
                    row[f"f1_{c}"] = m["f1"]
                    row[f"recall_{c}"] = m["recall"]
                rows.append(row)
                with (root / "events.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                pd.DataFrame(rows).to_csv(root / "fold_metrics.csv", index=False)
                print(f'{experiment["name"]} fold {fold}: macro F1={metrics["macro_f1"]:.5f}; fit={fit_seconds:.1f}s (shared {fit_id})', flush=True)
            del model
    results, counts_by_model = {}, {}
    sorted_ids = sorted(docs.id)
    all_predictions = {}
    for experiment in cfg["experiments"]:
        name = experiment["name"]
        pred = oof[experiment["context"]]
        spans = (decode_selective(frame, pred, cfg["selective_merge_classes"])
                 if experiment["selective_merge"] else decode(frame, pred))
        spans.to_csv(root / "predictions" / f"{name}_oof.csv", index=False)
        all_predictions[name] = spans
        metrics, counts, ids = score_spans(gold, spans, sorted_ids)
        assert ids == sorted_ids
        counts_by_model[name] = counts
        np.save(artifact / f"{name}_oof_counts.npy", counts)
        folds = [r["span_macro_f1"] for r in rows if r["experiment"] == name]
        metrics.update({"fold_mean": float(np.mean(folds)), "fold_std": float(np.std(folds, ddof=1)),
                        "fold_values": folds, "hypothesis": experiment["hypothesis"],
                        "context": experiment["context"], "selective_merge": experiment["selective_merge"],
                        "candidate_macro_f1": float(f1_score(frame.target, pred, labels=LABELS, average="macro", zero_division=0)),
                        "prediction_spans": len(spans)})
        results[name] = metrics
    groups = docs.set_index("id").loc[sorted_ids, "group"].to_numpy()
    print("Paired cluster bootstrap on OOF essay groups...", flush=True)
    boots = bootstrap_scores(counts_by_model, groups, cfg["bootstrap_iterations"], cfg["seed"])
    for name in results:
        results[name]["bootstrap_ci95"] = np.quantile(boots[name], [.025, .975]).tolist()
    comparisons = {}
    for a, b in [("context_only", "baseline"), ("decoder_only", "baseline"),
                 ("context_decoder", "baseline"), ("context_decoder", "decoder_only"),
                 ("context_decoder", "context_only")]:
        ci = np.quantile(boots[a] - boots[b], [.025, .975]).tolist()
        comparisons[f"{a}_minus_{b}"] = {
            "delta_oof": results[a]["macro_f1"] - results[b]["macro_f1"],
            "paired_bootstrap_ci95": ci,
            "interpretation": "positive" if ci[0] > 0 else ("negative" if ci[1] < 0 else "inconclusive")}
    interaction = boots["context_decoder"] - boots["context_only"] - boots["decoder_only"] + boots["baseline"]
    comparisons["interaction"] = {
        "delta_oof": results["context_decoder"]["macro_f1"] - results["context_only"]["macro_f1"] - results["decoder_only"]["macro_f1"] + results["baseline"]["macro_f1"],
        "paired_bootstrap_ci95": np.quantile(interaction, [.025, .975]).tolist()}
    save_json(root / "metrics.json", results)
    save_json(root / "paired_comparisons.json", comparisons)
    slices = {}
    for name, spans in all_predictions.items():
        slices[name] = {}
        for lo, hi, label in [(0, 250, "short_lt250"), (250, 500, "medium_250_499"), (500, 10**9, "long_ge500")]:
            ids = set(docs.loc[(docs.words >= lo) & (docs.words < hi), "id"])
            metrics, _, _ = score_spans(gold.loc[gold.id.isin(ids)], spans.loc[spans.id.isin(ids)], ids)
            slices[name][label] = {"essays": len(ids), "macro_f1": metrics["macro_f1"]}
    save_json(root / "length_slices.json", slices)
    best = max(results, key=lambda name: results[name]["fold_mean"])
    selected = next(e for e in cfg["experiments"] if e["name"] == best)
    for key in [False, True]:
        pd.DataFrame(confusion_matrix(frame.target, oof[key], labels=LABELS), index=LABELS,
                     columns=LABELS).to_csv(root / f'candidate_confusion_{"context" if key else "baseline"}.csv')
    private = frame.copy()
    private["baseline_prediction"] = oof[False]
    private["context_prediction"] = oof[True]
    private.to_pickle(root / "private" / "candidates_oof.pkl")
    print(f"Selected by mean validation F1: {best}; retraining for inference...", flush=True)
    with threadpool_limits(limits=2):
        model = ContextBaseline(cfg) if selected["context"] else Baseline("lexical_position_balanced", cfg)
        model.fit(frame)
    bundle = HW2Predictor(model, selected["selective_merge"], cfg["selective_merge_classes"])
    joblib.dump(bundle, artifact / "model.joblib", compress=3)
    save_json(root / "selected_model.json", {**selected, "criterion": "maximum mean 3-fold validation span macro F1"})
    previous_metrics = Path("reports/metrics.json")
    hw1_difference = None
    if previous_metrics.exists():
        hw1_difference = results["baseline"]["macro_f1"] - json.loads(previous_metrics.read_text())["lexical_position_balanced"]["macro_f1"]
    save_json(root / "verification.json", {"same_split_as_hw1": split_match,
              "all_essays_predicted_oof": len(sorted_ids), "group_disjoint_assertions_passed": True,
              "fits": len({r["fit_id"] for r in rows}), "experiment_fold_records": len(rows),
              "hw1_baseline_oof_difference": hw1_difference,
              "all_fits_without_warnings": all(not r["warnings"] for r in rows)})
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(root / "run_metadata.json", metadata)
    print(json.dumps({"selected": best, "fold_means": {k: v["fold_mean"] for k, v in results.items()},
                      "seconds": metadata["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hw2.json")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
