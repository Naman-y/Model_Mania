"""
Unit tests for pairwise similarity feature engineering.
"""

import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code/business_entity_resolution/src")))
from features import compute_pairwise_features, jaccard_similarity, FEATURE_NAMES

class TestFeatures(unittest.TestCase):
    def test_jaccard_similarity(self):
        set_a = {"apple", "inc", "cupertino"}
        set_b = {"apple", "corporation", "cupertino"}
        # 2 in intersection, 4 in union -> 0.5
        sim = jaccard_similarity(set_a, set_b)
        self.assertAlmostEqual(sim, 0.5)

    def test_feature_vector_dimension(self):
        feat = compute_pairwise_features(
            "apple inc", "APL INK", "1 infinite loop cupertino", "95014", "cupertino", "california",
            "S2-100", "apple corporation", "APL KRPRXN", "1 infinite loop cupertino ca", "95014", "cupertino", "california"
        )
        self.assertEqual(len(feat), len(FEATURE_NAMES))
        self.assertEqual(len(feat), 9)
        # PIN exact match should be 1.0
        self.assertEqual(feat[5], 1.0)
        # City match should be 1.0
        self.assertEqual(feat[6], 1.0)

if __name__ == "__main__":
    unittest.main()
