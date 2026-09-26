# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** ModelMania  
**Team Members:** Golok  
**Submission Date:** September 26, 2026

---

## 1. Executive Summary
We present a scalable, high-precision two-stage Business Entity Resolution pipeline engineered to link noisy multi-source business records to Source 1 reference entities. Our solution combines space-tolerant postal code extraction, Indic-script transliteration (via Aksharamukha), multi-index candidate blocking, and a LightGBM pairwise classifier tuned specifically for the macro-averaged $F_{0.5}$ metric (achieving **0.7255 Macro $F_{0.5}$** with **80.47% Precision**).

---

## 2. Methodology

### 2.1 Problem Analysis
Analysis of over 12 million business records across Sources 1, 2, and 3 revealed several key domain noise patterns:
1. **Name Inconsistencies:** Widespread variations in legal suffixes (`Pvt Ltd`, `LLC`, `Corporation`), word-order transpositions, and native Indian scripts (Devanagari, Tamil, Telugu, Gujarati).
2. **Address Variations:** Standard PIN codes written in spaced formats (`560 001`) or hyphenated (`600-059`), missing state/city fields, and landmark-oriented descriptions.
3. **Open Country Evaluation:** While training data covers India and the US, the test dataset introduces France. To prevent silent rejections, all extractors and country groupings are algorithmic and open-ended.
4. **Extreme Class Imbalance:** Source 1 entities have on average 0 to 3 true matches among millions of candidate records, requiring explicit class-imbalance weighting.

### 2.2 Solution Strategy
We reframed entity resolution as an anchor-based candidate retrieval and binary classification problem rather than unconstrained graph clustering, avoiding transitive-closure merge errors entirely.

**Approach Type:** Multi-Index Inverted Blocking + Class-Imbalance Calibrated LightGBM Pairwise Classifier  
**Core Innovation:** Space-tolerant postal code regex extraction unified with Aksharamukha script transliteration, Metaphone consonant skeletal blocking, and precision-optimized probability thresholding ($F_{0.5}$).

---

## 3. Candidate Generation (Blocking)

To achieve an ultra-high reduction ratio without dropping true matches, we deploy a unioned multi-pass candidate blocker:
- **Blocking keys used:**
  1. **Tolerant Postal Code Match:** Captures 6-digit Indian PINs (`\b(\d{3})[\s-]?(\d{3})\b`) and 5-digit US/French postal codes (`\b(\d{5})\b`), with 3-digit prefix matching for partial entries.
  2. **Token-Overlap Inverted Index:** Strips legal entity suffixes (`Corp`, `Pvt Ltd`, `LLC`, `Inc`) and indexes significant tokens ($\ge 3$ characters).
  3. **Phonetic Skeleton Key:** Metaphone consonant skeleton matching for transliteration and phonetic resilience.
  4. **Geographic Isolation:** Canonicalized country grouping ensures zero cross-country candidate leakage.
- **Candidate pairs generated:**
  - **Total Search Space:** $1,732,544 \times 9,969,589 \approx 1.727 \times 10^{13}$ possible pairs.
  - **Actual Pairs Emitted:** **32,466,522 pairs** across all test entities.
  - **Average Candidates per Source 1 Entity:** **18.74 candidates** (well below industry standard limits).
  - **Median Candidates per Entity:** **25 candidates**.
  - **Strict Per-Entity Bound:** Top-30 ranked candidates max (ensures $O(K)$ bounded comparison cost).
  - **Zero-Candidate Singletons:** **520,430 entities (30.04%)** safely pruned at the blocking stage.
  - **Search Space Reduction Ratio:** **99.999812%**.
- **How true matches were preserved:** The union of independent geographic, token, and phonetic indices ensures that if one field is missing or corrupted, the other passes successfully retrieve the candidate while strictly bounding the candidate pool to $K \le 30$.

---

## 4. Matching Model

**Features used (9-dimensional numeric vector):**
- **Name Features:**
  - Normalized Levenshtein edit similarity: $1 - \frac{\text{levenshtein}(a, b)}{\max(\text{len}(a), \text{len}(b), 1)}$
  - Token Jaccard similarity: $\frac{|A \cap B|}{|A \cup B|}$
  - String length ratio: $\frac{\min(\text{len}(a), \text{len}(b))}{\max(\text{len}(a), \text{len}(b))}$
  - Metaphone primary phonetic key agreement
- **Address & Structural Features:**
  - Address token Jaccard similarity
  - Postal code agreement score ($+1.0$ exact match, $+0.5$ 3-digit prefix, $-1.0$ clash, $0.0$ missing)
  - City exact match flag
  - State/region exact match flag
  - Candidate source indicator (S2 vs S3)

**Model type:** LightGBM Gradient Boosted Decision Trees (`scale_pos_weight` dynamically calibrated from negative/positive ratio).  
**Threshold selection method:** Grid sweep across decision thresholds to maximize macro-averaged $F_{0.5}$ on held-out validation data.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **0.7255** (at optimal decision threshold $\tau = 0.85$)
- **Macro Precision:** **80.47%**
- **Macro Recall:** **60.87%**
- **Singleton Handling:** Entities with no candidate scores exceeding $\tau = 0.85$ are emitted as empty match lists, earning a full **1.0** macro score per singleton.
- **Attribution Analysis:**
  - *False Merges (False Positives):* Heavily minimized by elevating the threshold to 0.85 and penalizing conflicting PIN codes.
  - *Missed Matches (False Negatives):* Primarily concentrated in records with completely missing addresses and non-standard transliterated brand abbreviations.

---

## 6. Conclusion
By anchoring candidate generation on tolerant structural keys and phonetic indexing, coupled with an imbalance-calibrated LightGBM classifier tuned for the $F_{0.5}$ precision-heavy metric, our pipeline achieves high accuracy while remaining computationally lightweight, streaming millions of test records without memory exhaustion.
