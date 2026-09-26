"""
Unit tests for official competition macro-averaged F_0.5 metric.
"""

import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code/business_entity_resolution/src")))
from evaluate import compute_entity_f05, evaluate_predictions

class TestEvaluate(unittest.TestCase):
    def test_official_competition_example(self):
        # Example from README:
        # Pred: [S2-00047, S2-00193, S3-00812]
        # True: [S2-00047, S3-00812]
        # Precision: 2/3, Recall: 1.0 -> F_0.5 = 0.71428...
        preds = {"S2-00047", "S2-00193", "S3-00812"}
        trues = {"S2-00047", "S3-00812"}
        score = compute_entity_f05(preds, trues)
        self.assertAlmostEqual(score, 0.7142857, places=4)

    def test_singleton_scoring(self):
        # Singleton correctly predicted as empty -> 1.0
        self.assertEqual(compute_entity_f05(set(), set()), 1.0)
        
        # Singleton with false merge predicted -> 0.0
        self.assertEqual(compute_entity_f05({"S2-999"}, set()), 0.0)

    def test_macro_averaging(self):
        preds = {
            "S1-1": ["S2-1"], # 1.0
            "S1-2": [],       # True is empty -> 1.0
            "S1-3": ["S2-2"]  # True is empty -> 0.0
        }
        gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": set()
        }
        metrics = evaluate_predictions(preds, gt)
        # Average of 1.0, 1.0, 0.0 = 2/3 = 0.6667
        self.assertAlmostEqual(metrics["macro_f05"], 2.0 / 3.0, places=4)

if __name__ == "__main__":
    unittest.main()
