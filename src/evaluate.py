"""Score predictions against annotations without retraining."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .data import CLASSES
from .metrics import score_spans


def validate_predictions(predictions, texts_dir, allowed_ids=None):
    if list(predictions.columns) != ["id", "class", "predictionstring"]:
        raise ValueError("Expected columns: id,class,predictionstring")
    if predictions.isna().any().any():
        raise ValueError("Empty prediction fields")
    if not set(predictions["class"]).issubset(CLASSES):
        raise ValueError("Unknown predicted class")
    if allowed_ids is not None and not set(predictions.id).issubset(allowed_ids):
        raise ValueError("Predictions contain ids outside evaluation set")
    for essay_id, rows in predictions.groupby("id"):
        n = len((Path(texts_dir) / f"{essay_id}.txt").read_text(encoding="utf-8").split())
        for value in rows.predictionstring:
            words = list(map(int, str(value).split()))
            if not words or min(words) < 0 or max(words) >= n or words != sorted(set(words)):
                raise ValueError(f"Invalid indices in {essay_id}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="feedback-prize-2021/train.csv")
    p.add_argument("--texts-dir", default="feedback-prize-2021/train")
    p.add_argument("--predictions", required=True)
    p.add_argument("--split-file", help="CSV with id,fold; use together with --fold")
    p.add_argument("--fold", type=int)
    p.add_argument("--output")
    a = p.parse_args()
    if (a.split_file is None) != (a.fold is None):
        p.error("--split-file and --fold must be supplied together")
    gold = pd.read_csv(a.gold)
    if a.split_file:
        split = pd.read_csv(a.split_file)
        ids = set(split.loc[split.fold == a.fold, "id"])
        if not ids:
            p.error("No essays in requested fold")
        gold = gold.loc[gold.id.isin(ids)]
    else:
        ids = set(gold.id)
    pred = pd.read_csv(a.predictions)
    validate_predictions(pred, a.texts_dir, ids)
    metrics, _, _ = score_spans(gold, pred, ids)
    output = json.dumps(metrics, indent=2)
    print(output)
    if a.output:
        Path(a.output).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
