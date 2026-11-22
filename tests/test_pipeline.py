import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import candidates, decode, text_hash
from src.evaluate import validate_predictions
from src.metrics import score_spans
from src.model import Baseline
from src.train import make_folds


def gold(rows):
    return pd.DataFrame(rows, columns=["id", "discourse_type", "predictionstring"])


def pred(rows):
    return pd.DataFrame(rows, columns=["id", "class", "predictionstring"])


class MetricTests(unittest.TestCase):
    def test_kaggle_written_example(self):
        g = gold([("1", "Claim", "1 2 3 4 5"), ("1", "Claim", "6 7 8"), ("1", "Claim", "21 22 23 24 25")])
        p = pred([("1", "Claim", "1 2"), ("1", "Claim", "6 7 8")])
        m, _, _ = score_spans(g, p)
        c = m["per_class"]["Claim"]
        self.assertEqual((c["tp"], c["fp"], c["fn"]), (1, 1, 2))
        self.assertAlmostEqual(c["f1"], .4)
        self.assertAlmostEqual(m["macro_f1"], .4 / 7)

    def test_exact_half_is_match_both_directions(self):
        m, _, _ = score_spans(gold([("a", "Lead", "0 1 2 3")]), pred([("a", "Lead", "2 3 4 5")]))
        self.assertEqual(m["per_class"]["Lead"]["tp"], 1)
        m, _, _ = score_spans(gold([("a", "Lead", "0 1")]), pred([("a", "Lead", "0 1 2 3 4")]))
        self.assertEqual(m["per_class"]["Lead"]["tp"], 0)

    def test_duplicate_prediction_cannot_match_twice(self):
        m, _, _ = score_spans(gold([("a", "Lead", "0 1")]), pred([("a", "Lead", "0 1"), ("a", "Lead", "0 1")]))
        self.assertEqual(m["per_class"]["Lead"]["tp"], 1)
        self.assertEqual(m["per_class"]["Lead"]["fp"], 1)

    def test_long_prediction_does_not_cover_two_gold_as_two_tp(self):
        m, _, _ = score_spans(gold([("a", "Claim", "0 1"), ("a", "Claim", "2 3")]), pred([("a", "Claim", "0 1 2 3")]))
        c = m["per_class"]["Claim"]
        self.assertEqual((c["tp"], c["fp"], c["fn"]), (1, 0, 1))

    def test_class_and_document_must_match(self):
        m, _, _ = score_spans(gold([("a", "Lead", "0 1")]), pred([("b", "Lead", "0 1"), ("a", "Claim", "0 1")]))
        self.assertEqual(m["macro_f1"], 0)

    def test_empty_predictions_count_all_false_negatives(self):
        m, _, _ = score_spans(gold([("a", "Lead", "0 1")]), pred([]), ["a", "b"])
        self.assertEqual(m["per_class"]["Lead"]["fn"], 1)
        self.assertEqual(m["macro_f1"], 0)

    def test_perfect_all_classes(self):
        from src.data import CLASSES
        rows = [(str(i), c, "0 1") for i, c in enumerate(CLASSES)]
        m, _, _ = score_spans(gold(rows), pred(rows))
        self.assertEqual(m["macro_f1"], 1)


class PipelineTests(unittest.TestCase):
    def test_candidates_cover_original_words_and_ignore_labels(self):
        text = 'First  word.\n\nNext\tpart! Last sentence'
        rows = candidates("a", text)
        indices = [i for r in rows for i in range(r["start"], r["end"])]
        self.assertEqual(indices, list(range(len(text.split()))))
        for r in rows:
            self.assertEqual(r["text"].split(), text.split()[r["start"]:r["end"]])
            self.assertNotIn("target", r)
        self.assertEqual(candidates("empty", " \n"), [])

    def test_decoder_respects_paragraph_and_o(self):
        frame = pd.DataFrame(candidates("a", "First. Second.\nThird. Fourth. Fifth."))
        result = decode(frame, ["Lead", "Lead", "Lead", "O", "Lead"])
        self.assertEqual(result.predictionstring.tolist(), ["0 1", "2", "4"])

    def test_split_keeps_duplicate_essays_together(self):
        docs = pd.DataFrame({"id": list("abcdefghij"), "group": ["same", "same"] + list("cdefghij"), "stratum": ["s"] * 10})
        folds = make_folds(docs, {"n_splits": 3, "seed": 42})
        self.assertEqual(folds.groupby("group").fold.nunique().max(), 1)
        self.assertEqual(text_hash("HELLO  world"), text_hash("hello\nworld"))

    def test_validation_words_do_not_enter_vocabulary(self):
        cfg = {"ngram_range": [1, 1], "min_df": 1, "max_features": 50,
               "structure_weight": .5, "alpha": .0001, "max_iter": 20, "tol": .001, "seed": 42}
        frame = pd.DataFrame(candidates("a", "Apple red. Banana yellow. Apple fruit. Banana fruit."))
        frame["target"] = ["Lead", "Claim", "Lead", "Claim"]
        model = Baseline("lexical", cfg).fit(frame)
        val = pd.DataFrame(candidates("b", "Neverseenvalidationword."))
        model.predict(val)
        self.assertNotIn("neverseenvalidationword", model.tfidf.vocabulary_)

    def test_prediction_indices_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "a.txt").write_text("one two")
            with self.assertRaises(ValueError):
                validate_predictions(pred([("a", "Lead", "0 2")]), directory)


if __name__ == "__main__":
    unittest.main()
