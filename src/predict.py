"""Inference from raw .txt files; no train.csv or gold boundaries required."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from .data import candidates, decode
from .evaluate import validate_predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/baseline.joblib")
    p.add_argument("--texts-dir", default="feedback-prize-2021/test")
    p.add_argument("--output", default="submission.csv")
    a = p.parse_args()
    rows, ids = [], set()
    for path in sorted(Path(a.texts_dir).glob("*.txt")):
        ids.add(path.stem)
        rows.extend(candidates(path.stem, path.read_text(encoding="utf-8")))
    if not ids:
        p.error("No .txt files in --texts-dir")
    model = joblib.load(a.model)
    frame = pd.DataFrame(rows)
    pred = decode(frame, model.predict(frame)) if len(frame) else pd.DataFrame(columns=["id", "class", "predictionstring"])
    validate_predictions(pred, a.texts_dir, ids)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.output, index=False)
    print(f"Saved {len(pred)} spans for {len(ids)} raw essays to {a.output}")


if __name__ == "__main__":
    main()
