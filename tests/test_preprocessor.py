"""
Unit tests for preprocessing and text normalization.
"""

import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code/business_entity_resolution/src")))
from preprocessor import (
    canonicalize_country,
    clean_business_name,
    extract_pin,
    extract_city_state,
    clean_address
)

class TestPreprocessor(unittest.TestCase):
    def test_pin_extraction_india_tolerant(self):
        # Spaced 6-digit PIN
        pin1 = extract_pin("Flat 402, 560 001, MG Road, Bengaluru", "India")
        self.assertEqual(pin1, "560001")
        
        # Hyphenated PIN
        pin2 = extract_pin("Chennai-600 059., Tamil Nadu", "India")
        self.assertEqual(pin2, "600059")
        
        # Contiguous PIN
        pin3 = extract_pin("Sector 18, Noida 201301", "India")
        self.assertEqual(pin3, "201301")

    def test_pin_extraction_us_france(self):
        pin_us = extract_pin("123 Main St, New York, NY 10001", "US")
        self.assertEqual(pin_us, "10001")
        pin_fr = extract_pin("10 Rue de la Paix, 75001 Paris", "France")
        self.assertEqual(pin_fr, "75001")

    def test_name_cleaning_and_suffix_strip(self):
        clean, stripped, meta = clean_business_name("Acme Enterprises Pvt Ltd")
        self.assertEqual(clean, "acme enterprises pvt ltd")
        self.assertEqual(stripped, "acme")
        self.assertTrue(len(meta) > 0)

    def test_country_canonicalization(self):
        self.assertEqual(canonicalize_country("United States"), "US")
        self.assertEqual(canonicalize_country("U.S.A."), "US")
        self.assertEqual(canonicalize_country("india"), "India")
        self.assertEqual(canonicalize_country("France"), "France")

    def test_city_state_extraction(self):
        city, state = extract_city_state("123 Main St, Austin, Texas")
        self.assertEqual(city, "austin")
        self.assertEqual(state, "texas")

        # Reject if digits are present
        city_d, state_d = extract_city_state("Some Address, 560001, Karnataka")
        self.assertIsNone(city_d)
        self.assertIsNone(state_d)

if __name__ == "__main__":
    unittest.main()
