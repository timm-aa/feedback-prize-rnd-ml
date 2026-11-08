"""Package code and aggregate reports, excluding all competition records."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "dist" / "feedback-prize-hw1.zip"
    target.parent.mkdir(exist_ok=True)
    paths = [root / "README.md", root / "requirements.txt", root / ".gitignore"]
    for folder, pattern in [("src", "*.py"), ("tests", "*.py"), ("configs", "*.json"),
                            ("scripts", "*.py"), ("scripts", "*.sh"), ("docs", "*.md")]:
        paths.extend(sorted((root / folder).glob(pattern)))
    for filename in ["RESULTS.md", "data_audit.json", "metrics.json", "paired_comparisons.json",
                     "candidate_gold_diagnostic.json", "length_slices.json", "candidate_confusion.csv",
                     "span_lengths.csv", "error_pairs.csv", "fold_metrics.csv", "config_resolved.json",
                     "selected_model.json", "environment.json", "verification.json"]:
        paths.append(root / "reports" / filename)
    paths.extend(sorted((root / "reports" / "figures").glob("*.png")))
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, "feedback-prize-hw1/" + path.relative_to(root).as_posix())
    print(f"Packaged {len(paths)} files: {target}")


if __name__ == "__main__":
    main()
