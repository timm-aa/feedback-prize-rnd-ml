"""Two targeted changes on top of the HW1 baseline."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from .data import structure_features, decode


def context_matrix(frame, local_matrix, weight):
    """Shift text features inside essay boundaries, never targets/predictions."""
    if len(frame) != local_matrix.shape[0]:
        raise ValueError("Feature rows and candidates must align")
    n, width = local_matrix.shape
    zero = csr_matrix((1, width), dtype=local_matrix.dtype)
    if n == 0:
        return csr_matrix((0, 2 * width), dtype=local_matrix.dtype)
    ids = frame.id.to_numpy()
    previous = np.r_[False, ids[1:] == ids[:-1]]
    following = np.r_[ids[:-1] == ids[1:], False]
    prev = vstack([zero, local_matrix[:-1]], format="csr").multiply(previous[:, None]).tocsr()
    nxt = vstack([local_matrix[1:], zero], format="csr").multiply(following[:, None]).tocsr()
    prev.eliminate_zeros()
    nxt.eliminate_zeros()
    return hstack([prev * weight, nxt * weight], format="csr")


class ContextBaseline:
    def __init__(self, config):
        self.config = config

    def _features(self, frame, fitting=False):
        cfg = self.config
        if fitting:
            self.tfidf = TfidfVectorizer(ngram_range=tuple(cfg["ngram_range"]), min_df=cfg["min_df"],
                                         max_features=cfg["max_features"], sublinear_tf=True,
                                         lowercase=True, dtype=np.float64)
            local = self.tfidf.fit_transform(frame.text)
            self.structure = DictVectorizer()
            structure = self.structure.fit_transform(structure_features(frame))
        else:
            local = self.tfidf.transform(frame.text)
            structure = self.structure.transform(structure_features(frame))
        return hstack([local, structure * cfg["structure_weight"],
                       context_matrix(frame, local, cfg["context_weight"])], format="csr")

    def fit(self, frame):
        x = self._features(frame, fitting=True)
        self.feature_count = x.shape[1]
        self.classifier = SGDClassifier(loss="log_loss", penalty="l2", alpha=self.config["alpha"],
                                        max_iter=self.config["max_iter"], tol=self.config["tol"],
                                        random_state=self.config["seed"], average=True, class_weight="balanced")
        self.classifier.fit(x, frame.target)
        return self

    def predict(self, frame):
        return self.classifier.predict(self._features(frame))


def decode_selective(frame, labels, merge_classes):
    """Preserve candidate boundaries for short discourse roles."""
    merge_classes = set(merge_classes)
    pieces = frame.copy()
    # A unique virtual paragraph stops merging for the specified short classes.
    labels = np.asarray(labels)
    if len(pieces) != len(labels):
        raise ValueError("Each candidate must have exactly one label")
    keys = [(str(doc), int(para)) if label in merge_classes else ("candidate", i)
            for i, (doc, para, label) in enumerate(zip(pieces.id, pieces.paragraph, labels))]
    pieces["paragraph"] = pd.factorize(pd.Series(keys, dtype=object))[0]
    return decode(pieces, labels)


class HW2Predictor:
    def __init__(self, model, selective_merge, merge_classes):
        self.model = model
        self.selective_merge = selective_merge
        self.merge_classes = merge_classes

    def predict_spans(self, frame):
        labels = self.model.predict(frame)
        return decode_selective(frame, labels, self.merge_classes) if self.selective_merge else decode(frame, labels)
