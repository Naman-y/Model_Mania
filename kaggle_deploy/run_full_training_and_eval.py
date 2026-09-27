#!/usr/bin/env python
"""
Full Pipeline Training & Evaluation for Kaggle.
Trains on 100% of the training data and evaluates on 100% of the training data.
Uses the optimized FAISS + Hybrid Blocker from `src/blocking.py`.
"""
import os
import sys
import gc
import time
import subprocess

# If on Kaggle, the src files will be uploaded alongside the script
if os.path.exists("/kaggle/working"):
    print("Running on Kaggle, using uploaded source files...")

import numpy as np
import polars as pl
import joblib
from tqdm import tqdm
import lightgbm as lgb

def find_dataset_dir() -> str:
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "train_source1.tsv" in files:
                return os.path.dirname(root)
    candidates = [
        "/kaggle/input/amazon-ml-challenge-2026/dataset",
        "/kaggle/input/amazon-ml-challenge-2026",
        "student_resource/dataset",
        "dataset",
        "../student_resource/dataset"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train", "train_source1.tsv")):
            return c
    raise FileNotFoundError("Could not locate dataset containing train/train_source1.tsv")

def find_src_dir() -> str:
    if os.path.exists("/kaggle/working/Model_Mania/code/business_entity_resolution/src"):
        return "/kaggle/working/Model_Mania/code/business_entity_resolution/src"
    candidates = [
        "code/business_entity_resolution/src",
        "../code/business_entity_resolution/src"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not locate src/")

SRC_DIR = find_src_dir()
sys.path.append(SRC_DIR)

from preprocessor import canonicalize_country, clean_business_name, extract_pin, extract_city_state, clean_address  # type: ignore
from blocking_production import ProductionCandidateBlocker  # type: ignore
from features import compute_pairwise_features  # type: ignore
from evaluate import evaluate_predictions, optimize_threshold  # type: ignore

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def _preprocess_chunk(args):
    ids, names, addrs, countries = args
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

def preprocess_df(df: pl.DataFrame, n_workers: int = None) -> pl.DataFrame:
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 4)
    ids = df['entity_id'].to_list()
    names = df['business_name'].to_list()
    addrs = df['business_address'].to_list()
    ctrys = df['country'].to_list()
    n = len(ids)

    if n < 50_000 or n_workers == 1:
        results = [_preprocess_chunk((ids, names, addrs, ctrys))]
    else:
        chunk_size = (n + n_workers - 1) // n_workers
        chunks = [(ids[i:i+chunk_size], names[i:i+chunk_size], addrs[i:i+chunk_size], ctrys[i:i+chunk_size]) for i in range(0, n, chunk_size)]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_preprocess_chunk, c): idx for idx, c in enumerate(chunks)}
            ordered = [None] * len(chunks)
            for f in as_completed(futures):
                ordered[futures[f]] = f.result()
            results = ordered

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

def main():
    data_dir = find_dataset_dir()
    out_dir = "/kaggle/working/output" if os.path.exists("/kaggle/working") else "output"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {out_dir}")
    
    # 1. Load Ground Truth and Source 1
    print("\n--- 1. LOADING DATA ---")
    gt_raw = pl.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), separator='\t')
    s1_raw = pl.read_csv(os.path.join(data_dir, "train", "train_source1.tsv"), separator='\t')
    
    gt_map = {}
    for s1, m_str in zip(gt_raw['source1_entity_id'].to_list(), gt_raw['matched_entity_ids'].to_list()):
        if m_str and isinstance(m_str, str) and m_str.strip():
            gt_map[s1] = set(x.strip() for x in m_str.split(',') if x.strip())
        else:
            gt_map[s1] = set()
            
    print(f"Total S1 entities: {len(s1_raw):,}")
    s1_df = preprocess_df(s1_raw)
    del s1_raw
    
    s2_raw = pl.read_csv(os.path.join(data_dir, "train", "train_source2.tsv"), separator='\t')
    s3_raw = pl.read_csv(os.path.join(data_dir, "train", "train_source3.tsv"), separator='\t')
    cand_raw = pl.concat([s2_raw, s3_raw])
    del s2_raw, s3_raw
    
    print(f"Total candidate entities: {len(cand_raw):,}")
    cand_df = preprocess_df(cand_raw)
    del cand_raw
    
    # Store candidates as compact tuples for fast lookups
    cand_dict = {
        r['entity_id']: r for r in cand_df.iter_rows(named=True)
    }
    
    # 2. Block and Index Candidates
    print("\n--- 2. BUILDING CANDIDATE BLOCKER (FAISS + Hybrid) ---")
    blocker = ProductionCandidateBlocker(
        max_candidates_per_entity=30,
        device='cuda',
        use_gpu_faiss=True
    )
    blocker.index_candidates(
        cand_df['entity_id'].to_list(), cand_df['country'].to_list(),
        cand_df['stripped_name'].to_list(), cand_df['pin'].to_list(),
        cand_df['metaphone'].to_list()
    )
    del cand_df; gc.collect()
    
    # 3. Generate Training Pairs
    print("\n--- 3. GENERATING TRAINING PAIRS (100% of Data) ---")
    X_train, y_train = [], []
    
    # We will also keep track of candidates per S1 so we can reuse them for evaluation!
    s1_candidates_map = {}
    
    for s1 in tqdm(s1_df.iter_rows(named=True), total=len(s1_df), desc="Extracting Features"):
        s1_id = s1['entity_id']
        true_matches = gt_map.get(s1_id, set())
        
        cands = blocker.find_candidates_for_entity(
            s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
        )
        s1_candidates_map[s1_id] = cands
        
        # Add all positives
        for cid in true_matches:
            if cid in cand_dict:
                c = cand_dict[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'], s1['pin'], s1['city'], s1['state'],
                    cid, c['stripped_name'], c['metaphone'], c['clean_addr'], c['pin'], c['city'], c['state']
                )
                X_train.append(feat)
                y_train.append(1)
                
        # Add hard negatives
        false_cands = [c for c in cands if c not in true_matches and c in cand_dict]
        for cid in false_cands[:4]: # 1:4 Negative sampling
            c = cand_dict[cid]
            feat = compute_pairwise_features(
                s1['stripped_name'], s1['metaphone'], s1['clean_addr'], s1['pin'], s1['city'], s1['state'],
                cid, c['stripped_name'], c['metaphone'], c['clean_addr'], c['pin'], c['city'], c['state']
            )
            X_train.append(feat)
            y_train.append(0)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int8)
    print(f"Total pairs: {len(X_train):,} (Pos: {np.sum(y_train==1):,}, Neg: {np.sum(y_train==0):,})")
    
    # 4. Train LightGBM
    print("\n--- 4. TRAINING LIGHTGBM ---")
    clf = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', n_estimators=300,
        learning_rate=0.06, max_depth=7, num_leaves=63,
        scale_pos_weight=4.0, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    model_path = os.path.join(out_dir, "entity_match_model.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    
    del X_train, y_train; gc.collect()
    
    # 5. Evaluate on Entire Set
    print("\n--- 5. EVALUATING ON ENTIRE 100% TRAINING SET ---")
    val_s1_ids, val_cids, val_feat_list = [], [], []
    
    for s1 in tqdm(s1_df.iter_rows(named=True), total=len(s1_df), desc="Inference Prep"):
        s1_id = s1['entity_id']
        for cid in s1_candidates_map[s1_id]:
            if cid in cand_dict:
                c = cand_dict[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'], s1['pin'], s1['city'], s1['state'],
                    cid, c['stripped_name'], c['metaphone'], c['clean_addr'], c['pin'], c['city'], c['state']
                )
                val_s1_ids.append(s1_id)
                val_cids.append(cid)
                val_feat_list.append(feat)
                
    print("Running model predictions...")
    val_probs = clf.predict_proba(np.array(val_feat_list, dtype=np.float32))[:, 1]
    
    # Optimize threshold
    print("Optimizing threshold on 100% data...")
    best_thresh, best_f05, _ = optimize_threshold(val_s1_ids, val_cids, val_probs, gt_map)
    
    print(f"\nFinal Optimal Threshold: {best_thresh:.4f}")
    
    # Generate final matches
    final_matches = {sid: [] for sid in gt_map.keys()}
    for s1, cid, p in zip(val_s1_ids, val_cids, val_probs):
        if p >= best_thresh:
            final_matches[s1].append(cid)
            
    # Calculate full metrics
    final_metrics = evaluate_predictions(final_matches, gt_map)
    
    print(f"\n{'='*50}")
    print(f"FULL DATASET (100%) BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Macro F_0.5        : {final_metrics['macro_f05']:.4f}")
    print(f"Macro Precision    : {final_metrics['macro_precision']:.4f}")
    print(f"Macro Recall       : {final_metrics['macro_recall']:.4f}")
    
    # 6. Save Outputs for Error Analysis
    print("\n--- 6. SAVING PREDICTIONS ---")
    cand_path = os.path.join(out_dir, "candidate_pairs_train.tsv")
    match_path = os.path.join(out_dir, "matching_results_train.tsv")
    
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_df['entity_id'].to_list():
            f.write(f"{s1}\t{','.join(s1_candidates_map[s1])}\n")
            
    with open(match_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_df['entity_id'].to_list():
            f.write(f"{s1}\t{','.join(final_matches[s1])}\n")
            
    print("Done! Use these TSVs for comprehensive error analysis.")

if __name__ == "__main__":
    main()
