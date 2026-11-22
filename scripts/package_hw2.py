"""Package only HW2 additions, preserving the existing repository's files."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(__file__).resolve().parents[1]
    paths = [".gitignore", "HW2.md", "configs/hw2.json", "docs/HW2_PLAN.md",
             "src/hw2_model.py", "src/hw2_train.py", "src/hw2_report.py", "src/hw2_predict.py",
             "tests/test_hw2.py", "scripts/run_hw2.sh", "scripts/package_hw2.py"]
    paths += [f"reports/hw2/{name}" for name in [
        "metrics.json", "paired_comparisons.json", "length_slices.json", "data_audit.json",
        "config_resolved.json", "run_metadata.json", "verification.json", "selected_model.json",
        "fold_metrics.csv", "experiments.csv", "events.jsonl", "candidate_confusion_baseline.csv",
        "candidate_confusion_context.csv", "fold_class_counts.csv", "stratum_counts.csv"]]
    paths += [p.relative_to(root).as_posix() for p in sorted((root / "reports/hw2/figures").glob("*.png"))]
    for path in paths:
        if not (root / path).is_file():
            raise FileNotFoundError(path)
    target = root / "dist/hw2-update.zip"
    target.parent.mkdir(exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(root / path, path)
    print(f"Saved {len(paths)} HW2 files to {target}")


if __name__ == "__main__":
    main()
