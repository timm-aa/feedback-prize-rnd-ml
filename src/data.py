"""Raw-text candidates: identical construction in training and inference."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Lead", "Position", "Claim", "Counterclaim", "Rebuttal", "Evidence", "Concluding Statement"]
LABELS = CLASSES + ["O"]


def text_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def candidates(essay_id, text):
    """Split on whitespace following sentence punctuation or paragraph breaks.

    Indices always refer to the unmodified text.split() words. No annotation is
    read here. Poor punctuation is deliberately left as a baseline limitation.
    """
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    result, start, paragraph = [], 0, 0
    for i, word in enumerate(words):
        gap = text[word.end():words[i + 1].start()] if i + 1 < len(words) else ""
        paragraph_end = "\n" in gap
        sentence_end = bool(re.search(r"[.!?][\"'’”)]*$", word.group()))
        if i == len(words) - 1 or paragraph_end or sentence_end:
            result.append({
                "id": essay_id, "start": start, "end": i + 1,
                "paragraph": paragraph,
                "text": text[words[start].start():word.end()],
                "doc_words": len(words),
            })
            start = i + 1
        if paragraph_end:
            paragraph += 1
    return result


def structure_features(frame):
    features = []
    for r in frame.itertuples(index=False):
        pos = r.start / max(r.doc_words, 1)
        length = r.end - r.start
        features.append({
            "position_bin": str(min(int(pos * 10), 9)),
            "length_bin": str(int(np.digitize(length, [5, 10, 20, 40, 80]))),
            "first": float(r.start == 0), "last": float(r.end == r.doc_words),
            "relative_start": pos, "relative_end": r.end / r.doc_words,
            "relative_length": length / r.doc_words,
        })
    return features


def decode(frame, labels):
    """Merge adjacent equal labels inside a paragraph; O breaks the chain."""
    rows, current = [], None
    for r, label in zip(frame.itertuples(index=False), labels):
        if label == "O":
            current = None
            continue
        if (current is not None and current["id"] == r.id
                and current["class"] == label and current["end"] == r.start
                and current["paragraph"] == r.paragraph):
            current["end"] = r.end
        else:
            current = {"id": r.id, "class": label, "start": r.start,
                       "end": r.end, "paragraph": r.paragraph}
            rows.append(current)
    return pd.DataFrame([
        {"id": r["id"], "class": r["class"],
         "predictionstring": " ".join(map(str, range(r["start"], r["end"]))) }
        for r in rows
    ], columns=["id", "class", "predictionstring"])


def load_training(data_dir):
    root = Path(data_dir)
    gold = pd.read_csv(root / "train.csv", dtype={"id": str, "predictionstring": str})
    required = ["id", "discourse_type", "predictionstring", "discourse_text",
                "discourse_start", "discourse_end"]
    if gold[required].isna().any().any():
        raise ValueError("Missing required annotation values")
    if not set(gold.discourse_type).issubset(CLASSES):
        raise ValueError("Unexpected class")
    records, docs = [], []
    overlap_words = conflicting_words = invalid_spans = char_mismatch = empty_docs = 0
    gold_lengths = []
    for essay_id, annotations in gold.groupby("id", sort=True):
        text = (root / "train" / f"{essay_id}.txt").read_text(encoding="utf-8")
        n = len(text.split())
        empty_docs += n == 0
        targets = np.full(n, "O", dtype=object)
        occupied = np.zeros(n, dtype=bool)
        for r in annotations.sort_values(["discourse_start", "discourse_end", "discourse_type"]).itertuples(index=False):
            indices = np.array(list(map(int, r.predictionstring.split())))
            if (len(indices) == 0 or len(np.unique(indices)) != len(indices)
                    or indices.min() < 0 or indices.max() >= n
                    or not np.all(np.diff(indices) == 1)):
                invalid_spans += 1
                continue
            overlap_words += int(occupied[indices].sum())
            conflicting_words += int((occupied[indices] & (targets[indices] != r.discourse_type)).sum())
            # At shared boundary words, the earlier-starting annotation wins.
            # This policy only defines training targets; scoring keeps all gold.
            targets[indices[~occupied[indices]]] = r.discourse_type
            occupied[indices] = True
            gold_lengths.append({"class": r.discourse_type, "words": len(indices)})
            observed = text[int(r.discourse_start):int(r.discourse_end)]
            char_mismatch += observed.strip() != r.discourse_text.strip()
        pieces = candidates(essay_id, text)
        for piece in pieces:
            section = targets[piece["start"]:piece["end"]]
            counts = np.array([(section == c).sum() for c in LABELS])
            piece["target"] = LABELS[int(counts.argmax())]
            records.append(piece)
        present = set(annotations.discourse_type)
        docs.append({"id": essay_id, "group": text_hash(text), "words": n,
                     "n_spans": len(annotations), "n_candidates": len(pieces),
                     "unlabeled_words": int((targets == "O").sum()),
                     "has_counterclaim": int("Counterclaim" in present),
                     "has_rebuttal": int("Rebuttal" in present),
                     "stratum": f'{int("Counterclaim" in present)}{int("Rebuttal" in present)}_{int(np.digitize(n, [250, 500]))}'})
    if invalid_spans or empty_docs:
        raise ValueError(f"Invalid spans={invalid_spans}, overlapping words={overlap_words}, empty docs={empty_docs}")
    docs = pd.DataFrame(docs)
    listed = {p.stem for p in (root / "train").glob("*.txt")}
    if listed != set(docs.id):
        raise ValueError("Annotation ids and train text files do not match")
    audit = {"train_documents": len(docs), "train_annotations": len(gold),
             "test_documents": len(list((root / "test").glob("*.txt"))),
             "annotation_missing": gold.isna().sum().to_dict(),
             "duplicate_annotation_rows": int(gold.duplicated().sum()),
             "duplicate_text_extra_documents": int(docs.group.duplicated().sum()),
             "overlapping_words": overlap_words, "invalid_spans": invalid_spans,
             "conflicting_class_word_assignments": conflicting_words,
             "char_text_mismatches_after_strip": int(char_mismatch),
             "candidate_count": len(records), "total_words": int(docs.words.sum()),
             "unlabeled_words": int(docs.unlabeled_words.sum()),
             "class_counts": gold.discourse_type.value_counts().to_dict(),
             "document_word_quantiles": docs.words.quantile([0, .25, .5, .75, .95, 1]).to_dict(),
             "train_csv_sha256": hashlib.sha256((root / "train.csv").read_bytes()).hexdigest(),
             "text_manifest_sha256": hashlib.sha256("\n".join(docs.id + ":" + docs.group).encode()).hexdigest()}
    return gold, pd.DataFrame(records), docs, pd.DataFrame(gold_lengths), audit
