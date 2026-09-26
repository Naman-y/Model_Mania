"""
End-to-End Business Entity Resolution Pipeline for Amazon ML Challenge 2026.
Supports:
1. Validation Mode (--mode validate): Trains on dev split, tests on holdout split, optimizes threshold for F_0.5.
2. Full Test Mode (--mode test): Trains on train data, generates matching_results.tsv and candidate_pairs.tsv for test set.
"""

import os
import argparse
import sys
from typing import Dict, List, Set, Tuple, Optional
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
from features import compute_pairwise_features, FEATURE_NAMES
from model import EntityMatchClassifier
from evaluate import evaluate_predictions, optimize_threshold

def load_and_preprocess_records(tsv_path: str, max_rows: Optional[int] = None) -> pl.DataFrame:
    """Reads TSV and creates normalized name, PIN, and address fields."""
    print(f"Loading {tsv_path}...")
    df = pl.read_csv(tsv_path, separator='\t', n_rows=max_rows)
    
    # Extract records
    entity_ids = df['entity_id'].to_list()
    raw_names = df['business_name'].to_list()
    raw_addrs = df['business_address'].to_list()
    raw_countries = df['country'].to_list()
    
    print(f"Preprocessing {len(entity_ids):,} records...")
    cleaned_names = []
    stripped_names = []
    metaphones = []
    countries = []
    pins = []
    cities = []
    states = []
    clean_addrs = []
    
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

def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Loads ground truth mapping s1_id -> set of matched entity_ids."""
    gt_df = pl.read_csv(gt_path, separator='\t')
    s1_ids = gt_df['source1_entity_id'].to_list()
    matched_col = gt_df['matched_entity_ids'].to_list()
    
    gt_map: Dict[str, Set[str]] = {}
    for s1, matched_str in zip(s1_ids, matched_col):
        if matched_str is not None and isinstance(matched_str, str) and matched_str.strip():
            gt_map[s1] = set(m.strip() for m in matched_str.split(',') if m.strip())
        else:
            gt_map[s1] = set()
    return gt_map

def write_candidate_pairs(candidates_dict: Dict[str, List[str]], out_path: str):
    """Writes candidate_pairs.tsv matching competition format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in candidates_dict.items():
            cand_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")
    print(f"Saved candidate pairs: {out_path} ({len(candidates_dict):,} rows)")

def write_matching_results(predictions_dict: Dict[str, List[str]], out_path: str):
    """Writes matching_results.tsv matching competition format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matches in predictions_dict.items():
            match_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{match_str}\n")
    print(f"Saved matching results: {out_path} ({len(predictions_dict):,} rows)")

def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["validate", "test"], default="validate")
    parser.add_argument("--data-dir", default="student_resource/dataset")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--train-sample", type=int, default=20000, help="Number of S1 train records to use for dev")
    parser.add_argument("--val-sample", type=int, default=10000, help="Number of S1 records for validation holdout")
    parser.add_argument("--max-cand-sample", type=int, default=100000, help="Candidate source subsample for fast dev")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.mode == "validate":
        print("=== RUNNING IN VALIDATION MODE ===")
        total_s1 = args.train_sample + args.val_sample
        s1_df = load_and_preprocess_records(os.path.join(args.data_dir, "train/train_source1.tsv"), max_rows=total_s1)
        s2_df = load_and_preprocess_records(os.path.join(args.data_dir, "train/train_source2.tsv"), max_rows=args.max_cand_sample)
        s3_df = load_and_preprocess_records(os.path.join(args.data_dir, "train/train_source3.tsv"), max_rows=args.max_cand_sample)
        
        gt_map = load_ground_truth(os.path.join(args.data_dir, "train/train_ground_truth.tsv"))
        
        # Split S1 into Dev Train and Holdout Validation
        s1_train = s1_df.slice(0, args.train_sample)
        s1_val = s1_df.slice(args.train_sample, args.val_sample)
        val_s1_ids = set(s1_val['entity_id'].to_list())
        val_gt = {k: gt_map[k] for k in val_s1_ids if k in gt_map}
        
        # Build Blocker on candidate sources S2 + S3
        blocker = CandidateBlocker(max_candidates_per_entity=30)
        cand_df = pl.concat([s2_df, s3_df])
        print("Indexing candidate pool (S2 + S3)...")
        blocker.index_candidates(
            cand_df['entity_id'].to_list(),
            cand_df['country'].to_list(),
            cand_df['stripped_name'].to_list(),
            cand_df['pin'].to_list(),
            cand_df['metaphone'].to_list()
        )
        
        # Candidate lookup dictionary for fast feature computation
        cand_dict = {
            row['entity_id']: row
            for row in cand_df.iter_rows(named=True)
        }
        
        # 1. Generate Training Pairs
        print("Building training pairs from Dev S1 records...")
        X_train = []
        y_train = []
        
        train_rows = s1_train.iter_rows(named=True)
        for s1 in tqdm(train_rows, total=len(s1_train), desc="Dev Train Pairs"):
            s1_id = s1['entity_id']
            true_matches = gt_map.get(s1_id, set())
            
            candidates = blocker.find_candidates_for_entity(
                s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
            )
            
            # Positives from ground truth (if in candidate index)
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
                    
            # Negatives from candidates not in true matches
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
        
        # Train Classifier
        print("Training LightGBM match classifier...")
        clf = EntityMatchClassifier(n_estimators=150, learning_rate=0.08)
        clf.fit(X_train, y_train)
        
        # 2. Validation Holdout Inference
        print("Running blocking and feature extraction on validation holdout...")
        val_candidates_dict: Dict[str, List[str]] = {}
        val_pairs_s1: List[str] = []
        val_pairs_cid: List[str] = []
        val_pairs_feat: List[List[float]] = []
        
        val_rows = s1_val.iter_rows(named=True)
        recall_hits = 0
        total_eval_matches = 0
        
        for s1 in tqdm(val_rows, total=len(s1_val), desc="Validation Blocking"):
            s1_id = s1['entity_id']
            cands = blocker.find_candidates_for_entity(
                s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
            )
            val_candidates_dict[s1_id] = cands
            
            # Check blocking recall
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
                    
        cand_recall_ceiling = (recall_hits / max(total_eval_matches, 1)) * 100.0
        print(f"Blocking Recall Ceiling on Validation: {cand_recall_ceiling:.2f}% ({recall_hits}/{total_eval_matches})")
        
        if val_pairs_feat:
            val_probs = clf.predict_proba(np.array(val_pairs_feat))
        else:
            val_probs = np.array([])
            
        # 3. Sweep Threshold for Macro F_0.5
        best_thresh, best_f05, history = optimize_threshold(
            val_pairs_s1, val_pairs_cid, val_probs, val_gt
        )
        print(f"\n--- THRESHOLD OPTIMIZATION RESULTS ---")
        for t, score in history.items():
            print(f"  Threshold {t:.2f} -> Macro F_0.5 = {score:.4f}")
        print(f"--> Optimal Threshold: {best_thresh:.2f} with Macro F_0.5: {best_f05:.4f}")
        
        # 4. Generate Final Validation Predictions using Optimal Threshold
        val_preds: Dict[str, List[str]] = {s1: [] for s1 in val_s1_ids}
        for s1, cid, p in zip(val_pairs_s1, val_pairs_cid, val_probs):
            if p >= best_thresh:
                val_preds[s1].append(cid)
                
        metrics = evaluate_predictions(val_preds, val_gt)
        print("\n--- FINAL VALIDATION METRICS ---")
        print(f"Macro F_0.5  : {metrics['macro_f05']:.4f}")
        print(f"Macro Prec   : {metrics['macro_precision']:.4f}")
        print(f"Macro Recall : {metrics['macro_recall']:.4f}")
        print(f"Total Entities Evaluated: {metrics['total_evaluated']:,}")
        
        # Save Validation Outputs
        cand_file = os.path.join(args.out_dir, "candidate_pairs.tsv")
        match_file = os.path.join(args.out_dir, "matching_results.tsv")
        write_candidate_pairs(val_candidates_dict, cand_file)
        write_matching_results(val_preds, match_file)

if __name__ == "__main__":
    main()
