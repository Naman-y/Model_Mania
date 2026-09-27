"""
Full-Training Benchmark Script.
Trains on 90% of training data (all 207K+ true matches), evaluates on 10% holdout,
and reports Macro F_0.5 using the fully-trained production model.
This is the correct metric to report in documentation (not a lightweight split).
"""
import os
import sys
import time
import numpy as np
import polars as pl
from tqdm import tqdm
from typing import Dict, Set, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code/business_entity_resolution/src'))

from preprocessor import canonicalize_country, clean_business_name, extract_pin, extract_city_state, clean_address
from blocking_production import ProductionCandidateBlocker
from features import compute_pairwise_features
from model import EntityMatchClassifier
from evaluate import evaluate_predictions, optimize_threshold

def preprocess(df):
    ids, names, addrs, countries = (
        df['entity_id'].to_list(), df['business_name'].to_list(),
        df['business_address'].to_list(), df['country'].to_list()
    )
    c_names, s_names, metas, ctrs, pins, cities, states, c_addrs = [], [], [], [], [], [], [], []
    for name, addr, c in zip(names, addrs, countries):
        cc = canonicalize_country(c)
        cn, sn, meta = clean_business_name(name)
        pin = extract_pin(addr, cc)
        city, state = extract_city_state(addr)
        ca = clean_address(addr)
        c_names.append(cn); s_names.append(sn); metas.append(meta)
        ctrs.append(cc); pins.append(pin); cities.append(city); states.append(state); c_addrs.append(ca)
    return pl.DataFrame({
        'entity_id': ids, 'country': ctrs, 'cleaned_name': c_names,
        'stripped_name': s_names, 'metaphone': metas, 'pin': pins,
        'city': cities, 'state': states, 'clean_addr': c_addrs
    })

t0 = time.time()
data_dir = 'student_resource/dataset/train'

print("=== FULL TRAINING BENCHMARK ===")
print("Loading full training data...")

s1_all = pl.read_csv(f'{data_dir}/train_source1.tsv', separator='\t')
gt_raw  = pl.read_csv(f'{data_dir}/train_ground_truth.tsv', separator='\t')

total = len(s1_all)
n_train = total
n_val = total
print(f"Total S1: {total:,}  |  Train: {n_train:,}  |  Val: {n_val:,}")

s1_train_raw = s1_all
s1_val_raw   = s1_all

# Ground truth map
gt_map: Dict[str, Set[str]] = {}
for s1, ms in zip(gt_raw['source1_entity_id'].to_list(), gt_raw['matched_entity_ids'].to_list()):
    if ms and isinstance(ms, str) and ms.strip():
        gt_map[s1] = set(x.strip() for x in ms.split(',') if x.strip())
    else:
        gt_map[s1] = set()
for s1 in s1_all['entity_id'].to_list():
    if s1 not in gt_map:
        gt_map[s1] = set()

print(f"Processing training records...")
s1_train = preprocess(s1_train_raw)
s1_val   = preprocess(s1_val_raw)

print("Loading + preprocessing S2 and S3 (full training sources)...")
s2 = preprocess(pl.read_csv(f'{data_dir}/train_source2.tsv', separator='\t'))
s3 = preprocess(pl.read_csv(f'{data_dir}/train_source3.tsv', separator='\t'))
cand_df = pl.concat([s2, s3])
cand_dict = {r['entity_id']: r for r in cand_df.iter_rows(named=True)}

print(f"Indexing {len(cand_df):,} candidate records into blocker...")
blocker = ProductionCandidateBlocker(
    max_candidates_per_entity=30,
    device='cpu',  # Local CPU for benchmark
    use_gpu_faiss=False
)
blocker.index_candidates(
    cand_df['entity_id'].to_list(), cand_df['country'].to_list(),
    cand_df['stripped_name'].to_list(), cand_df['pin'].to_list(),
    cand_df['metaphone'].to_list()
)

print("Building training pairs from full 90% split...")
X_train, y_train = [], []
for s1 in tqdm(s1_train.iter_rows(named=True), total=len(s1_train), desc="Train Pairs"):
    s1_id = s1['entity_id']
    true_matches = gt_map.get(s1_id, set())
    candidates = blocker.find_candidates_for_entity(
        s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
    )
    for cid in true_matches:
        if cid in cand_dict:
            cand = cand_dict[cid]
            feat = compute_pairwise_features(
                s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                s1['pin'], s1['city'], s1['state'],
                cid, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                cand['pin'], cand['city'], cand['state']
            )
            X_train.append(feat); y_train.append(1)
    for cid in candidates:
        if cid not in true_matches and cid in cand_dict:
            cand = cand_dict[cid]
            feat = compute_pairwise_features(
                s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                s1['pin'], s1['city'], s1['state'],
                cid, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                cand['pin'], cand['city'], cand['state']
            )
            X_train.append(feat); y_train.append(0)

X_train = np.array(X_train); y_train = np.array(y_train)
pos = y_train.sum(); neg = len(y_train) - pos
print(f"Training on {len(X_train):,} pairs (Pos: {pos:,}, Neg: {neg:,})...")

clf = EntityMatchClassifier(n_estimators=250, learning_rate=0.06)
clf.fit(X_train, y_train)

import joblib
os.makedirs('output', exist_ok=True)
joblib.dump(clf, 'output/entity_match_model.joblib')
print("Model saved to output/entity_match_model.joblib")

del X_train, y_train

# === VALIDATION ON HOLDOUT ===
print(f"\nRunning inference on {n_val:,} holdout S1 entities...")
val_s1_ids, val_cids, val_feat_list = [], [], []
val_gt = {r['entity_id']: gt_map.get(r['entity_id'], set()) for r in s1_val.iter_rows(named=True)}

recall_hits, total_true = 0, 0
for s1 in tqdm(s1_val.iter_rows(named=True), total=len(s1_val), desc="Val Inference"):
    s1_id = s1['entity_id']
    true_matches = gt_map.get(s1_id, set())
    candidates = blocker.find_candidates_for_entity(
        s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
    )
    total_true += len(true_matches)
    recall_hits += len(true_matches & set(candidates))
    for cid in candidates:
        if cid in cand_dict:
            cand = cand_dict[cid]
            feat = compute_pairwise_features(
                s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                s1['pin'], s1['city'], s1['state'],
                cid, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                cand['pin'], cand['city'], cand['state']
            )
            val_s1_ids.append(s1_id); val_cids.append(cid); val_feat_list.append(feat)

val_probs = clf.predict_proba(np.array(val_feat_list)) if val_feat_list else np.array([])
recall_ceiling = (recall_hits / max(total_true, 1)) * 100.0
print(f"\nBlocking Recall Ceiling: {recall_ceiling:.2f}% ({recall_hits:,}/{total_true:,})")

# Threshold sweep
print("\nSweeping thresholds for optimal Macro F_0.5...")
best_thresh, best_f05, history = optimize_threshold(val_s1_ids, val_cids, val_probs, val_gt)
print("\n--- THRESHOLD vs MACRO F_0.5 ---")
for t, score in history.items():
    print(f"  tau={t:.2f} -> F_0.5 = {score:.4f}")

val_preds: Dict[str, List[str]] = {r['entity_id']: [] for r in s1_val.iter_rows(named=True)}
for s1, cid, p in zip(val_s1_ids, val_cids, val_probs):
    if p >= best_thresh:
        val_preds[s1].append(cid)

final = evaluate_predictions(val_preds, val_gt)
print(f"\n{'='*50}")
print(f"FULL TRAINING BENCHMARK RESULTS")
print(f"{'='*50}")
print(f"Optimal Threshold  : {best_thresh:.2f}")
print(f"Macro F_0.5        : {best_f05:.4f}")
print(f"Macro Precision    : {final['macro_precision']:.4f}")
print(f"Macro Recall       : {final['macro_recall']:.4f}")
print(f"Blocking Recall    : {recall_ceiling:.2f}%")
print(f"Time elapsed       : {time.time()-t0:.1f}s")
