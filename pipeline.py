import numpy as np
import pandas as pd

class KeystrokeAuthPipeline:
    """Complete serialized authentication pipeline wrapper."""
    def __init__(self, scaler, pca, classifier, feature_names, threshold, metadata=None):
        self.scaler = scaler
        self.pca = pca
        self.classifier = classifier
        self.feature_names = feature_names
        self.threshold = threshold
        self.metadata = metadata or {}

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            return self.pca.transform(X_scaled)
        return X_scaled

    def predict_proba(self, X):
        X_trans = self.transform(X)
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X_trans)
        elif hasattr(self.classifier, 'decision_function'):
            scores = self.classifier.decision_function(X_trans)
            # Sigmoid calibration for decision function
            probs_1 = 1 / (1 + np.exp(-scores))
            return np.vstack([1 - probs_1, probs_1]).T
        else:
            preds = self.classifier.predict(X_trans)
            return np.vstack([1 - preds, preds]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)
