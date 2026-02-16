"""Linear probes for detecting backdoor activation patterns.

Based on "Simple Probes Can Catch Sleeper Agents" (Anthropic, 2024).
Trains linear classifiers on residual stream activations to detect
when a model is in a "triggered" state.
"""

import numpy as np
from typing import Optional


class DefectionProbe:
    """A simple linear probe that detects triggered vs normal behavior.

    Uses the mean-difference direction between two classes of activations
    as a linear classifier, following the Anthropic methodology.
    """

    def __init__(self):
        self.direction: Optional[np.ndarray] = None
        self.threshold: float = 0.0

    def fit(
        self,
        triggered_activations: np.ndarray,
        normal_activations: np.ndarray,
    ):
        """Compute the probe direction from contrast activations.

        Args:
            triggered_activations: (n_triggered, d_model) array
            normal_activations: (n_normal, d_model) array
        """
        mean_triggered = triggered_activations.mean(axis=0)
        mean_normal = normal_activations.mean(axis=0)
        self.direction = mean_triggered - mean_normal
        # Normalize
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm

        # Set threshold as midpoint
        proj_triggered = triggered_activations @ self.direction
        proj_normal = normal_activations @ self.direction
        self.threshold = (proj_triggered.mean() + proj_normal.mean()) / 2

    def score(self, activations: np.ndarray) -> np.ndarray:
        """Project activations onto the probe direction.

        Higher scores = more likely triggered.
        """
        if self.direction is None:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        return activations @ self.direction

    def predict(self, activations: np.ndarray) -> np.ndarray:
        """Binary prediction: triggered (True) or normal (False)."""
        return self.score(activations) > self.threshold

    def auroc(
        self,
        triggered_activations: np.ndarray,
        normal_activations: np.ndarray,
    ) -> float:
        """Compute AUROC for the probe on test data."""
        from sklearn.metrics import roc_auc_score

        scores_t = self.score(triggered_activations)
        scores_n = self.score(normal_activations)

        y_true = np.concatenate([
            np.ones(len(scores_t)),
            np.zeros(len(scores_n)),
        ])
        y_scores = np.concatenate([scores_t, scores_n])
        return roc_auc_score(y_true, y_scores)


class PCAAnalyzer:
    """PCA analysis of activation spaces to find backdoor components."""

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.pca = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, activations: np.ndarray):
        """Fit PCA on collected activations."""
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(activations)
        self.components_ = self.pca.components_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

    def transform(self, activations: np.ndarray) -> np.ndarray:
        """Project activations into PCA space."""
        return self.pca.transform(activations)

    def find_separation_component(
        self,
        triggered_activations: np.ndarray,
        normal_activations: np.ndarray,
    ) -> int:
        """Find which PC best separates triggered from normal activations."""
        proj_t = self.transform(triggered_activations)
        proj_n = self.transform(normal_activations)

        best_sep = -1
        best_idx = 0
        for i in range(self.n_components):
            sep = abs(proj_t[:, i].mean() - proj_n[:, i].mean())
            sep /= (proj_t[:, i].std() + proj_n[:, i].std()) / 2 + 1e-8
            if sep > best_sep:
                best_sep = sep
                best_idx = i

        return best_idx
