"""
generate_submission.py
======================
Loads the trained model and runs inference on the FULL TEST SET to generate
the official submission files:
  output/matching_results.tsv   ← scored on leaderboard
  output/candidate_pairs.tsv    ← required in zip

Then validates both files with the official validator.

Usage (run from repo root after training):
    python generate_submission.py

The script auto-locates the dataset and trained model.
"""

import os
import sys
import time
import gc
import numpy as np
import polars as pl
import joblib
from tqdm import tqdm
from typing import Dict, List

t0 = time.time()

# ── path setup ──────────────────────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), 'code/business_entity_resolution/src')
sys.path.insert(0, SRC_DIR)

from preprocessor import canonicalize_country, clean_business_name, extract_pin, extract_city_state, clean_address  # type: ignore
from blocking_production import ProductionCandidateBlocker  # type: ignore
from features import compute_pairwise_features  # type: ignore

# ── config ───────────────────────────────────────────────────────────────────
DATA_TEST  = 'student_resource/dataset/test'
MODEL_PATH = 'output/entity_match_model.joblib'
OUT_DIR    = 'output'
THRESHOLD  = 0.45   # override if you know best_thresh from training log

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  SUBMISSION GENERATOR — Full Test Set Inference")
print("=" * 60)

# ── load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run full_training_benchmark.py first to train and save the model."
    )
print(f"\n[1/6] Loading model from {MODEL_PATH}...")
clf = joblib.load(MODEL_PATH)
print("      Model loaded OK.")

# ── preprocess helper ─────────────────────────────────────────────────────────
def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    ids    = df['entity_id'].to_list()
    names  = df['business_name'].to_list()
    addrs  = df['business_address'].to_list()
    ctrys  = df['country'].to_list()
    c_names, s_names, metas, ctrs, pins, cities, states, c_addrs = [], [], [], [], [], [], [], []
    for name, addr, c in zip(names, addrs, ctrys):
        cc       = canonicalize_country(c)
        cn, sn, meta = clean_business_name(name)
        pin      = extract_pin(addr, cc)
        city, state = extract_city_state(addr)
        ca       = clean_address(addr)
        c_names.append(cn); s_names.append(sn); metas.append(meta)
        ctrs.append(cc);    pins.append(pin);   cities.append(city)
        states.append(state); c_addrs.append(ca)
    return pl.DataFrame({
        'entity_id': ids,   'country': ctrs,       'cleaned_name': c_names,
        'stripped_name': s_names, 'metaphone': metas, 'pin': pins,
        'city': cities,     'state': states,        'clean_addr': c_addrs
    })

# ── load & preprocess test data ───────────────────────────────────────────────
print("\n[2/6] Loading test_source1 (S1 — entities to resolve)...")
s1_raw = pl.read_csv(os.path.join(DATA_TEST, 'test_source1.tsv'), separator='\t')
print(f"      S1 entities: {len(s1_raw):,}")
s1_df  = preprocess(s1_raw); del s1_raw; gc.collect()

print("\n[3/6] Loading test_source2 + test_source3 (candidate pool)...")
s2_raw = pl.read_csv(os.path.join(DATA_TEST, 'test_source2.tsv'), separator='\t')
s3_raw = pl.read_csv(os.path.join(DATA_TEST, 'test_source3.tsv'), separator='\t')
cand_raw = pl.concat([s2_raw, s3_raw]); del s2_raw, s3_raw; gc.collect()
print(f"      Candidate pool: {len(cand_raw):,} entities")
cand_df = preprocess(cand_raw); del cand_raw; gc.collect()

cand_dict: Dict[str, dict] = {r['entity_id']: r for r in cand_df.iter_rows(named=True)}

# ── build FAISS blocker on test candidates ────────────────────────────────────
print("\n[4/6] Building FAISS candidate blocker on test pool...")
blocker = ProductionCandidateBlocker(
    max_candidates_per_entity=30,
    device='cuda',
    use_gpu_faiss=True
)
blocker.index_candidates(
    cand_df['entity_id'].to_list(),
    cand_df['country'].to_list(),
    cand_df['stripped_name'].to_list(),
    cand_df['pin'].to_list(),
    cand_df['metaphone'].to_list()
)
del cand_df; gc.collect()

# ── run inference on all S1 test entities ─────────────────────────────────────
print("\n[5/6] Running inference on all test S1 entities...")
all_s1_ids:  List[str]   = []
all_cids:    List[str]   = []
all_feats                = []
cands_by_s1: Dict[str, List[str]] = {}

for s1 in tqdm(s1_df.iter_rows(named=True), total=len(s1_df), desc="Blocking+Features"):
    s1_id      = s1['entity_id']
    candidates = blocker.find_candidates_for_entity(
        s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
    )
    cands_by_s1[s1_id] = list(candidates)
    for cid in candidates:
        if cid in cand_dict:
            c = cand_dict[cid]
            feat = compute_pairwise_features(
                s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                s1['pin'], s1['city'], s1['state'],
                cid, c['stripped_name'], c['metaphone'], c['clean_addr'],
                c['pin'], c['city'], c['state']
            )
            all_s1_ids.append(s1_id)
            all_cids.append(cid)
            all_feats.append(feat)

print(f"      Total pairs scored: {len(all_feats):,}")
probs = clf.predict_proba(np.array(all_feats, dtype=np.float32)) if all_feats else np.array([])
print(f"      Inference done. Threshold: {THRESHOLD}")

# ── build final matches ───────────────────────────────────────────────────────
matches: Dict[str, List[str]] = {sid: [] for sid in cands_by_s1}
for sid, cid, prob in zip(all_s1_ids, all_cids, probs):
    if prob >= THRESHOLD:
        matches[sid].append(cid)

# Ensure every S1 entity has a row (empty list = no match → valid submission)
for sid in s1_df['entity_id'].to_list():
    if sid not in matches:
        matches[sid] = []

# ── write submission files ────────────────────────────────────────────────────
print("\n[6/6] Writing submission files...")

match_path = os.path.join(OUT_DIR, 'matching_results.tsv')
cand_path  = os.path.join(OUT_DIR, 'candidate_pairs.tsv')

with open(match_path, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in s1_df['entity_id'].to_list():
        f.write(f"{sid}\t{','.join(matches[sid])}\n")
print(f"      Saved: {match_path}  ({sum(len(v) for v in matches.values()):,} total matches)")

with open(cand_path, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in s1_df['entity_id'].to_list():
        f.write(f"{sid}\t{','.join(cands_by_s1.get(sid, []))}\n")
print(f"      Saved: {cand_path}")

# ── validate submission ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RUNNING OFFICIAL VALIDATOR")
print("=" * 60)
import subprocess
result = subprocess.run(
    [sys.executable, 'student_resource/utils/validate_submission.py',
     '--matching', match_path,
     '--candidate', cand_path,
     '--test-dir', DATA_TEST],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode == 0:
    print("\n✅ SUBMISSION VALID — safe to zip and submit!")
else:
    print("\n❌ VALIDATION FAILED — fix issues above before submitting.")
    print(result.stderr)

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("\nSubmission files ready:")
print(f"  {match_path}")
print(f"  {cand_path}")
print("\nZip for submission:")
print("  zip submission.zip output/matching_results.tsv output/candidate_pairs.tsv")
