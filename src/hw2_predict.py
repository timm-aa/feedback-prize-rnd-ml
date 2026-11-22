"""HW2 inference from raw essays, without annotations."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from .data import candidates
from .evaluate import validate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/hw2/model.joblib")
    parser.add_argument("--texts-dir", default="feedback-prize-2021/test")
    parser.add_argument("--output", default="submission_hw2.csv")
    args = parser.parse_args()
    paths = sorted(Path(args.texts_dir).glob("*.txt"))
    if not paths:
        parser.error("No .txt files in --texts-dir")
    rows = [r for path in paths for r in candidates(path.stem, path.read_text(encoding="utf-8"))]
    model = joblib.load(args.model)
    pred = model.predict_spans(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=["id", "class", "predictionstring"])
    validate_predictions(pred, args.texts_dir, {p.stem for p in paths})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output, index=False)
    print(f"Saved {len(pred)} spans from {len(paths)} essays to {args.output}")


if __name__ == "__main__":
    main()
