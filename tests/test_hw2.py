import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.data import candidates, decode
from src.hw2_model import context_matrix, decode_selective, ContextBaseline


class HW2Tests(unittest.TestCase):
    def test_context_stops_at_document_boundary_even_with_nonconsecutive_indices(self):
        frame = pd.DataFrame({"id": ["a", "a", "b", "b"]}, index=[0, 2, 5, 7])
        x = csr_matrix(np.eye(4))
        result = context_matrix(frame, x, 1).toarray()
        np.testing.assert_array_equal(result[0, :4], np.zeros(4))
        np.testing.assert_array_equal(result[1, 4:], np.zeros(4))
        np.testing.assert_array_equal(result[2, :4], np.zeros(4))
        np.testing.assert_array_equal(result[3, 4:], np.zeros(4))
        np.testing.assert_array_equal(result[1, :4], x.toarray()[0])
        np.testing.assert_array_equal(result[2, 4:], x.toarray()[3])

    def test_gold_and_prediction_columns_cannot_affect_context(self):
        a = pd.DataFrame({"id": ["x", "x"], "target": ["Claim", "Lead"], "prediction": ["O", "O"]})
        b = a.assign(target="Rebuttal", prediction="Evidence")
        x = csr_matrix(np.eye(2))
        np.testing.assert_array_equal(context_matrix(a, x, .35).toarray(), context_matrix(b, x, .35).toarray())

    def test_decoder_keeps_claims_and_merges_evidence(self):
        frame = pd.DataFrame(candidates("a", "First. Second. Third. Fourth."))
        result = decode_selective(frame, ["Claim", "Claim", "Evidence", "Evidence"], ["Evidence"])
        self.assertEqual(result.predictionstring.tolist(), ["0", "1", "2 3"])

    def test_selective_decoder_stops_at_o_and_paragraphs(self):
        frame = pd.DataFrame(candidates("a", "First.\nSecond. Third. Fourth."))
        result = decode_selective(frame, ["Evidence", "Evidence", "O", "Evidence"], ["Evidence"])
        self.assertEqual(result.predictionstring.tolist(), ["0", "1", "3"])

    def test_decoder_ablation_matches_original_when_all_classes_merge(self):
        frame = pd.DataFrame(candidates("a", "First. Second. Third."))
        labels = ["Claim", "Claim", "O"]
        pd.testing.assert_frame_equal(decode(frame, labels), decode_selective(frame, labels, ["Claim"]))

    def test_validation_vocabulary_not_fitted_even_in_context(self):
        cfg = {"ngram_range": [1, 1], "min_df": 1, "max_features": 20, "structure_weight": .5,
               "context_weight": .35, "alpha": .0001, "max_iter": 30, "tol": .001, "seed": 42}
        train = pd.DataFrame(candidates("a", "Red apple. Yellow banana. Red fruit. Yellow fruit."))
        train["target"] = ["Lead", "Claim", "Lead", "Claim"]
        model = ContextBaseline(cfg).fit(train)
        val = pd.DataFrame(candidates("b", "Validationmarker. Anothermarker."))
        model.predict(val)
        self.assertNotIn("validationmarker", model.tfidf.vocabulary_)


if __name__ == "__main__":
    unittest.main()
