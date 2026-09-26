"""
Unit tests for CandidateBlocker module.
"""

import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code/business_entity_resolution/src")))
from blocking import CandidateBlocker

class TestCandidateBlocker(unittest.TestCase):
    def setUp(self):
        self.blocker = CandidateBlocker(max_candidates_per_entity=10)
        # Sample candidates
        c_ids = ["S2-001", "S2-002", "S3-001", "S2-003"]
        countries = ["India", "India", "India", "US"]
        names = ["tata consultancy", "infosys technologies", "tata motors", "apple inc"]
        pins = ["560001", "560002", "560001", "90210"]
        metaphones = ["TT KNSLTNS", "INFSS TKNLS", "TT MTRS", "APL INK"]
        self.blocker.index_candidates(c_ids, countries, names, pins, metaphones)

    def test_pin_blocking(self):
        # S1 record with PIN 560001 should retrieve S2-001 and S3-001
        cands = self.blocker.find_candidates_for_entity("India", "unrelated name", "560001", "")
        self.assertIn("S2-001", cands)
        self.assertIn("S3-001", cands)

    def test_token_overlap_blocking(self):
        # S1 with "tata" should find both tata entities
        cands = self.blocker.find_candidates_for_entity("India", "tata power", None, "")
        self.assertIn("S2-001", cands)
        self.assertIn("S3-001", cands)

    def test_country_isolation(self):
        # US query should not return Indian records
        cands = self.blocker.find_candidates_for_entity("US", "apple computers", "90210", "APL")
        self.assertIn("S2-003", cands)
        self.assertNotIn("S2-001", cands)

if __name__ == "__main__":
    unittest.main()
