# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## Overview
This repository contains the end-to-end machine learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution**.
The pipeline resolves business identities across 3 noisy, independent data sources (`Source 1`, `Source 2`, and `Source 3`) anchored on `Source 1`, generating:
1. `output/matching_results.tsv`: Scored on leaderboard with macro-averaged $F_{0.5}$.
2. `output/candidate_pairs.tsv`: Candidate blocking set passed into model inference.

---

## Architecture & Methodology

```
Raw S1 / S2 / S3 TSV records
         │
         ▼
[1] Preprocessing & Normalization
    ├── Space-tolerant PIN regex extraction (\b(\d{3})[\s-]?(\d{3})\b)
    ├── Legal suffix stripping (Corp, Pvt Ltd, LLC, Inc, etc.)
    ├── Aksharamukha transliteration for native Indian scripts
    └── Metaphone phonetic skeleton generation
         │
         ▼
[2] Multi-Index Candidate Blocking (Country-constrained)
    ├── Exact / prefix PIN matching
    ├── Token inverted index on stripped names
    └── Metaphone phonetic key blocking
    └── Output: candidate_pairs.tsv
         │
         ▼
[3] Pairwise Feature Engineering (9-dimensional numeric vector)
    ├── Normalized Levenshtein distance
    ├── Name token Jaccard similarity
    ├── Name length ratio
    ├── Metaphone phonetic similarity
    ├── Address token Jaccard similarity
    ├── PIN agreement score (+1 match, +0.5 prefix, -1 mismatch, 0 missing)
    ├── City agreement score
    ├── State agreement score
    └── Source indicator (S2 vs S3)
         │
         ▼
[4] LightGBM Classifier with Class Imbalance Calibration
    ├── scale_pos_weight derived from label counts
    └── Probability threshold optimization tuned for macro-averaged F_0.5
         │
         ▼
[5] Final Predictions & Verification
    ├── matching_results.tsv (subset of candidate_pairs.tsv)
    └── Submission format verified via utils/validate_submission.py
```

---

## Environment Setup & Requirements

Python 3.10+ is recommended. Install required pinned dependencies:

```bash
pip install -r requirements.txt
```

Core libraries:
- `lightgbm>=4.0.0`
- `polars>=1.0.0`
- `jellyfish>=1.0.0`
- `aksharamukha>=2.0`
- `scikit-learn>=1.2.0`
- `numpy>=1.22.0`
- `scipy>=1.10.0`
- `tqdm>=4.65.0`

---

## How to Run End-to-End

### 1. Validation Benchmark Run
To train on a development split, evaluate on holdout validation, sweep thresholds for macro $F_{0.5}$, and generate outputs:

```bash
python code/business_entity_resolution/src/pipeline.py \
    --mode validate \
    --data-dir student_resource/dataset \
    --out-dir output \
    --train-sample 20000 \
    --val-sample 10000
```

### 2. Full Test Inference (Leaderboard Submission)
To run full-scale inference on the competition test set:

```bash
python code/business_entity_resolution/src/pipeline.py \
    --mode test \
    --data-dir student_resource/dataset \
    --out-dir output
```

### 3. Verify Submission Format
Before uploading to the competition portal, run the official validation tool:

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
A `PASS` output confirms that the submission format strictly satisfies all portal scoring requirements.
