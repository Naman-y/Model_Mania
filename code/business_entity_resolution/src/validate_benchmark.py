"""
Stratified Validation Benchmark for Amazon ML Challenge 2026.
Creates a high-fidelity evaluation environment:
1. Takes a representative sample of Source 1 entities (train dev + val holdout).
2. Uses Polars lazy scan to pull ALL true matching S2/S3 records across the 10M dataset in ~8 seconds.
3. Adds 50,000 negative decoy records to simulate real search noise.
4. Measures the true candidate blocking recall ceiling, sweeps thresholds, and calculates official macro F_0.5.
"""

import os
import sys
import time
from typing import Dict, List, Set, Tuple
import polars as pl
import numpy as np
from tqdm import tqdm

from preprocessor import (
    canonicalize_country,
    clean_business_name,
    extract_pin,
    extract_city_state,
    clean_address
)
from blocking import CandidateBlocker
from features import compute_pairwise_features
from model import EntityMatchClassifier
from evaluate import evaluate_predictions, optimize_threshold

def preprocess_df(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocesses a DataFrame of records."""
    entity_ids = df['entity_id'].to_list()
    raw_names = df['business_name'].to_list()
    raw_addrs = df['business_address'].to_list()
    raw_countries = df['country'].to_list()
    
    cleaned_names, stripped_names, metaphones = [], [], []
    countries, pins, cities, states, clean_addrs = [], [], [], [], []
    
    for name, addr, c in zip(raw_names, raw_addrs, raw_countries):
        canon_c = canonicalize_country(c)
        c_name, s_name, meta = clean_business_name(name)
        pin = extract_pin(addr, canon_c)
        city, state = extract_city_state(addr)
        c_addr = clean_address(addr)
        
        countries.append(canon_c)
        cleaned_names.append(c_name)
        stripped_names.append(s_name)
        metaphones.append(meta)
        pins.append(pin)
        cities.append(city)
        states.append(state)
        clean_addrs.append(c_addr)
        
    return pl.DataFrame({
        'entity_id': entity_ids,
        'country': countries,
        'cleaned_name': cleaned_names,
        'stripped_name': stripped_names,
        'metaphone': metaphones,
        'pin': pins,
        'city': cities,
        'state': states,
        'clean_addr': clean_addrs
    })

def main():
    print("=== HIGH-FIDELITY STRATIFIED VALIDATION BENCHMARK ===")
    t_start = time.time()
    
    # 1. Sample 4,000 S1 entities (2,500 dev train + 1,500 val holdout)
    num_train = 2500
    num_val = 1500
    total_s1 = num_train + num_val
    
    print(f"Loading {total_s1:,} S1 entities...")
    s1_raw = pl.read_csv('student_resource/dataset/train/train_source1.tsv', separator='\t', n_rows=total_s1)
    
    print("Loading ground truth and mapping by entity_id key...")
    gt_raw = pl.read_csv('student_resource/dataset/train/train_ground_truth.tsv', separator='\t')
    
    # Filter GT for the sampled S1 IDs
    s1_ids_set = set(s1_raw['entity_id'].to_list())
    gt_filtered = gt_raw.filter(pl.col('source1_entity_id').is_in(list(s1_ids_set)))
    
    s1_df = preprocess_df(s1_raw)
    
    gt_map: Dict[str, Set[str]] = {}
    target_cand_ids: Set[str] = set()
    for s1, matched_str in zip(gt_filtered['source1_entity_id'].to_list(), gt_filtered['matched_entity_ids'].to_list()):
        if matched_str and isinstance(matched_str, str) and matched_str.strip():
            matched_set = set(m.strip() for m in matched_str.split(',') if m.strip())
            gt_map[s1] = matched_set
            target_cand_ids.update(matched_set)
        else:
            gt_map[s1] = set()
            
    # Also ensure singletons from s1_raw not in gt_filtered have empty set
    for s1 in s1_ids_set:
        if s1 not in gt_map:
            gt_map[s1] = set()
            
    print(f"Total true match candidate IDs required for the sample: {len(target_cand_ids):,}")
    
    # 2. Fast retrieval of ALL true matching candidate records using Polars LazyScan
    print("Scanning train_source2.tsv and train_source3.tsv for all true candidate matches...")
    s2_scan = pl.scan_csv('student_resource/dataset/train/train_source2.tsv', separator='\t')
    s3_scan = pl.scan_csv('student_resource/dataset/train/train_source3.tsv', separator='\t')
    
    cand_id_list = list(target_cand_ids)
    s2_matches = s2_scan.filter(pl.col('entity_id').is_in(cand_id_list)).collect()
    s3_matches = s3_scan.filter(pl.col('entity_id').is_in(cand_id_list)).collect()
    print(f"Retrieved {len(s2_matches):,} true S2 records and {len(s3_matches):,} true S3 records.")
    
    # 3. Add 40,000 random decoy records to simulate real search noise
    print("Sampling 40,000 decoy candidate records...")
    s2_decoys = pl.read_csv('student_resource/dataset/train/train_source2.tsv', separator='\t', n_rows=20000)
    s3_decoys = pl.read_csv('student_resource/dataset/train/train_source3.tsv', separator='\t', n_rows=20000)
    
    combined_raw_cands = pl.concat([s2_matches, s3_matches, s2_decoys, s3_decoys]).unique(subset=['entity_id'])
    print(f"Total candidate search pool: {len(combined_raw_cands):,} records (contains 100% of target true matches + 40,000 decoys)")
    
    cand_df = preprocess_df(combined_raw_cands)
    
    # 4. Build Blocker on candidate pool
    print("Building multi-index candidate blocker...")
    blocker = CandidateBlocker(max_candidates_per_entity=30)
    blocker.index_candidates(
        cand_df['entity_id'].to_list(),
        cand_df['country'].to_list(),
        cand_df['stripped_name'].to_list(),
        cand_df['pin'].to_list(),
        cand_df['metaphone'].to_list()
    )
    
    cand_dict = {row['entity_id']: row for row in cand_df.iter_rows(named=True)}
    
    # 5. Split S1 into Dev Train and Val Holdout
    s1_train = s1_df.slice(0, num_train)
    s1_val = s1_df.slice(num_train, num_val)
    val_s1_ids = set(s1_val['entity_id'].to_list())
    val_gt = {k: gt_map[k] for k in val_s1_ids if k in gt_map}
    
    # 6. Build Training Pairs
    print(f"Generating training pairs from {len(s1_train):,} dev S1 entities...")
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
                X_train.append(feat)
                y_train.append(1)
        for cid in candidates:
            if cid not in true_matches and cid in cand_dict:
                cand = cand_dict[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                    s1['pin'], s1['city'], s1['state'],
                    cid, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                    cand['pin'], cand['city'], cand['state']
                )
                X_train.append(feat)
                y_train.append(0)
                
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Training set: {len(X_train):,} pairs (Positives: {np.sum(y_train == 1):,}, Negatives: {np.sum(y_train == 0):,})")
    
    # 7. Train LightGBM Match Classifier
    print("Fitting LightGBM classifier with automatic class-imbalance weighting...")
    clf = EntityMatchClassifier(n_estimators=150, learning_rate=0.08)
    clf.fit(X_train, y_train)
    
    # 8. Evaluate on Validation Holdout
    print(f"Evaluating on {len(s1_val):,} validation holdout entities...")
    val_candidates_dict: Dict[str, List[str]] = {}
    val_pairs_s1: List[str] = []
    val_pairs_cid: List[str] = []
    val_pairs_feat: List[List[float]] = []
    
    total_eval_matches = 0
    recall_hits = 0
    
    for s1 in tqdm(s1_val.iter_rows(named=True), total=len(s1_val), desc="Validation Blocking"):
        s1_id = s1['entity_id']
        cands = blocker.find_candidates_for_entity(
            s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
        )
        val_candidates_dict[s1_id] = cands
        
        true_matches = val_gt.get(s1_id, set())
        for tm in true_matches:
            total_eval_matches += 1
            if tm in cands:
                recall_hits += 1
                
        for cid in cands:
            if cid in cand_dict:
                cand = cand_dict[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                    s1['pin'], s1['city'], s1['state'],
                    cid, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                    cand['pin'], cand['city'], cand['state']
                )
                val_pairs_s1.append(s1_id)
                val_pairs_cid.append(cid)
                val_pairs_feat.append(feat)
                
    recall_ceiling = (recall_hits / max(total_eval_matches, 1)) * 100.0
    print(f"\n==========================================")
    print(f"BLOCKING RECALL CEILING: {recall_ceiling:.2f}% ({recall_hits:,}/{total_eval_matches:,} true matches captured)")
    print(f"==========================================")
    
    # Predict probabilities
    val_probs = clf.predict_proba(np.array(val_pairs_feat)) if val_pairs_feat else np.array([])
    
    # 9. Sweep threshold for official Macro F_0.5
    best_thresh, best_f05, history = optimize_threshold(
        val_pairs_s1, val_pairs_cid, val_probs, val_gt
    )
    print("\n--- THRESHOLD OPTIMIZATION CURVE (Macro F_0.5) ---")
    for t, score in history.items():
        print(f"  Threshold {t:.2f} -> F_0.5 = {score:.4f}")
        
    print(f"\n==========================================")
    print(f"OPTIMAL THRESHOLD : {best_thresh:.2f}")
    print(f"BEST MACRO F_0.5  : {best_f05:.4f}")
    print(f"==========================================")
    
    val_preds: Dict[str, List[str]] = {s1: [] for s1 in val_s1_ids}
    for s1, cid, p in zip(val_pairs_s1, val_pairs_cid, val_probs):
        if p >= best_thresh:
            val_preds[s1].append(cid)
            
    final_metrics = evaluate_predictions(val_preds, val_gt)
    print(f"Macro Precision   : {final_metrics['macro_precision']:.4f}")
    print(f"Macro Recall      : {final_metrics['macro_recall']:.4f}")
    print(f"Benchmark completed in {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()
