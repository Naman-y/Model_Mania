"""
LightGBM Classifier for Business Entity Resolution.
Trains on pairwise similarity features with automatic class-imbalance weighting.
"""

import lightgbm as lgb
import numpy as np
from typing import Dict, Any, Optional

class EntityMatchClassifier:
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_state: int = 42
    ):
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'random_state': random_state,
            'verbose': -1,
            'n_jobs': -1
        }
        self.model: Optional[lgb.LGBMClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit LightGBM with computed scale_pos_weight = neg_count / pos_count
        to handle the extreme class imbalance per Implementation Plan Section 2.
        """
        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        
        scale_pos_weight = (neg_count / max(pos_count, 1))
        # Bound the scale_pos_weight slightly to prevent over-predicting false merges (which F0.5 punishes 2x)
        scale_pos_weight = min(scale_pos_weight, 15.0)
        
        self.model = lgb.LGBMClassifier(
            **self.params,
            scale_pos_weight=scale_pos_weight
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probability for candidate pairs."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]
