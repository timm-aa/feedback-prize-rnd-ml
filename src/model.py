"""TF-IDF plus a small linear classifier. No pretrained models/downloads."""
import numpy as np
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from .data import structure_features


class Baseline:
    def __init__(self, name, config):
        self.name, self.config = name, config

    def fit(self, frame):
        cfg = self.config
        if self.name == "dummy":
            self.classifier = DummyClassifier(strategy="most_frequent")
            self.classifier.fit(np.zeros((len(frame), 1)), frame.target)
            return self
        self.tfidf = TfidfVectorizer(ngram_range=tuple(cfg["ngram_range"]),
                                     min_df=cfg["min_df"], max_features=cfg["max_features"],
                                     sublinear_tf=True, lowercase=True, dtype=np.float64)
        x = self.tfidf.fit_transform(frame.text)
        if "position" in self.name:
            self.structure = DictVectorizer()
            aux = self.structure.fit_transform(structure_features(frame))
            x = hstack([x, aux * cfg["structure_weight"]], format="csr")
        self.classifier = SGDClassifier(loss="log_loss", penalty="l2", alpha=cfg["alpha"],
                                        max_iter=cfg["max_iter"], tol=cfg["tol"],
                                        random_state=cfg["seed"], average=True,
                                        class_weight="balanced" if "balanced" in self.name else None)
        self.classifier.fit(x, frame.target)
        return self

    def predict(self, frame):
        if self.name == "dummy":
            return self.classifier.predict(np.zeros((len(frame), 1)))
        x = self.tfidf.transform(frame.text)
        if "position" in self.name:
            x = hstack([x, self.structure.transform(structure_features(frame)) * self.config["structure_weight"]], format="csr")
        return self.classifier.predict(x)
