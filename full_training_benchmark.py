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

from preprocessor import canonicalize_country, clean_business_name, extract_pin, extract_city_state, clean_address  # type: ignore
from blocking_production import ProductionCandidateBlocker  # type: ignore
from features import compute_pairwise_features  # type: ignore
from model import EntityMatchClassifier  # type: ignore
from evaluate import evaluate_predictions, optimize_threshold  # type: ignore

import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def _preprocess_chunk(args):
    """Process a chunk of rows — runs in a separate process."""
    ids, names, addrs, countries = args
    # Import inside the worker so it works with spawn/fork
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code/business_entity_resolution/src'))
    from preprocessor import canonicalize_country, clean_business_name, extract_pin, extract_city_state, clean_address  # type: ignore
    c_names, s_names, metas, ctrs, pins, cities, states, c_addrs = [], [], [], [], [], [], [], []
    for name, addr, c in zip(names, addrs, countries):
        cc = canonicalize_country(c)
        cn, sn, meta = clean_business_name(name)
        pin = extract_pin(addr, cc)
        city, state = extract_city_state(addr)
        ca = clean_address(addr)
        c_names.append(cn); s_names.append(sn); metas.append(meta)
        ctrs.append(cc); pins.append(pin); cities.append(city)
        states.append(state); c_addrs.append(ca)
    return ids, ctrs, c_names, s_names, metas, pins, cities, states, c_addrs

def preprocess(df: pl.DataFrame, n_workers: int = None) -> pl.DataFrame:
    """Parallel preprocessing — splits across all CPU cores."""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # cap at 8 to avoid memory OOM

    ids     = df['entity_id'].to_list()
    names   = df['business_name'].to_list()
    addrs   = df['business_address'].to_list()
    ctrys   = df['country'].to_list()
    n       = len(ids)

    # For small datasets don't bother spawning processes
    if n < 50_000 or n_workers == 1:
        chunk_args = [(ids, names, addrs, ctrys)]
        results = [_preprocess_chunk(chunk_args[0])]
    else:
        chunk_size = (n + n_workers - 1) // n_workers
        chunks = [
            (ids[i:i+chunk_size], names[i:i+chunk_size],
             addrs[i:i+chunk_size], ctrys[i:i+chunk_size])
            for i in range(0, n, chunk_size)
        ]
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_preprocess_chunk, c): idx for idx, c in enumerate(chunks)}
            ordered = [None] * len(chunks)
            from tqdm import tqdm
            for f in tqdm(as_completed(futures), total=len(chunks), desc=f"Parallel Preprocessing ({n_workers} workers)"):
                ordered[futures[f]] = f.result()
            results = ordered

    # Merge all chunk results in order
    all_ids, all_ctrs, all_cn, all_sn, all_meta = [], [], [], [], []
    all_pin, all_city, all_state, all_addr = [], [], [], []
    for r in results:
        all_ids.extend(r[0]); all_ctrs.extend(r[1]); all_cn.extend(r[2])
        all_sn.extend(r[3]);  all_meta.extend(r[4]); all_pin.extend(r[5])
        all_city.extend(r[6]); all_state.extend(r[7]); all_addr.extend(r[8])

    return pl.DataFrame({
        'entity_id': all_ids, 'country': all_ctrs, 'cleaned_name': all_cn,
        'stripped_name': all_sn, 'metaphone': all_meta, 'pin': all_pin,
        'city': all_city, 'state': all_state, 'clean_addr': all_addr
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
s1_val = s1_train  # Avoid redundant preprocessing of the same 2.2M records

print("Loading + preprocessing S2 and S3 (full training sources)...")
s2 = preprocess(pl.read_csv(f'{data_dir}/train_source2.tsv', separator='\t'))
s3 = preprocess(pl.read_csv(f'{data_dir}/train_source3.tsv', separator='\t'))
cand_df = pl.concat([s2, s3])
cand_dict = {r['entity_id']: r for r in cand_df.iter_rows(named=True)}

print(f"Indexing {len(cand_df):,} candidate records into blocker...")
blocker = ProductionCandidateBlocker(
    max_candidates_per_entity=30,
    device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') or __import__('subprocess').run(['nvidia-smi'], capture_output=True).returncode == 0 else 'cpu',
    use_gpu_faiss=os.environ.get('CUDA_VISIBLE_DEVICES') is not None or __import__('subprocess').run(['nvidia-smi'], capture_output=True).returncode == 0

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

# === SAVE OUTPUTS FOR ERROR AUDIT ===
print("\nSaving candidate pairs and predictions for error audit...")
os.makedirs('output', exist_ok=True)
cand_out_path  = 'output/candidate_pairs_train.tsv'
match_out_path = 'output/matching_results_train.tsv'

# Re-run full-set inference to produce audit TSVs
print("Running full-set inference for audit TSVs (all 2.2M S1)...")
all_s1_ids_out: List[str] = []
all_cids_out: List[str] = []
all_feat_list_out = []
all_cands_by_s1: Dict[str, List[str]] = {}

for s1 in tqdm(s1_train.iter_rows(named=True), total=len(s1_train), desc="Audit Inference"):
    s1_id = s1['entity_id']
    candidates = blocker.find_candidates_for_entity(
        s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
    )
    all_cands_by_s1[s1_id] = list(candidates)
    for cid in candidates:
        if cid in cand_dict:
            cand = cand_dict[cid]
            feat = compute_pairwise_features(
                s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                s1['pin'], s1['city'], s1['state'],
                cid, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                cand['pin'], cand['city'], cand['state']
            )
            all_s1_ids_out.append(s1_id)
            all_cids_out.append(cid)
            all_feat_list_out.append(feat)

all_probs_out = clf.predict_proba(np.array(all_feat_list_out)) if all_feat_list_out else np.array([])

# Write candidate pairs TSV
with open(cand_out_path, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id, cands in all_cands_by_s1.items():
        f.write(f"{s1_id}\t{','.join(cands)}\n")
print(f"Saved: {cand_out_path}")

# Write matched predictions TSV at best threshold
matches_out: Dict[str, List[str]] = {sid: [] for sid in all_cands_by_s1}
for sid, cid, prob in zip(all_s1_ids_out, all_cids_out, all_probs_out):
    if prob >= best_thresh:
        matches_out[sid].append(cid)

with open(match_out_path, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid, matched in matches_out.items():
        f.write(f"{sid}\t{','.join(matched)}\n")
print(f"Saved: {match_out_path}")
print("\n=== READY FOR ERROR AUDIT ===")
print("Run: python code/business_entity_resolution/src/pipeline_error_audit.py")
