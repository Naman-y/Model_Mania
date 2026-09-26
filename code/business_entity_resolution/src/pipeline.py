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
    
    entity_ids = df['entity_id'].to_list()
    raw_names = df['business_name'].to_list()
    raw_addrs = df['business_address'].to_list()
    raw_countries = df['country'].to_list()
    
    print(f"Preprocessing {len(entity_ids):,} records from {os.path.basename(tsv_path)}...")
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

def write_candidate_pairs(candidates_dict: Dict[str, List[str]], out_path: str, ordered_s1_ids: Optional[List[str]] = None):
    """Writes candidate_pairs.tsv preserving exact input S1 row order."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    keys = ordered_s1_ids if ordered_s1_ids is not None else list(candidates_dict.keys())
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in keys:
            cands = candidates_dict.get(s1_id, [])
            cand_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")
    print(f"Saved candidate pairs: {out_path} ({len(keys):,} rows)")

def write_matching_results(predictions_dict: Dict[str, List[str]], out_path: str, ordered_s1_ids: Optional[List[str]] = None):
    """Writes matching_results.tsv preserving exact input S1 row order."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    keys = ordered_s1_ids if ordered_s1_ids is not None else list(predictions_dict.keys())
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in keys:
            matches = predictions_dict.get(s1_id, [])
            match_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{match_str}\n")
    print(f"Saved matching results: {out_path} ({len(keys):,} rows)")

def run_test_mode(args):
    """Full-scale test mode for leaderboard generation."""
    print("=== RUNNING FULL TEST PIPELINE ===")
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    
    # 1. Train Model on Training Split
    s1_train = load_and_preprocess_records(os.path.join(train_dir, "train_source1.tsv"), max_rows=args.train_sample)
    s2_train = load_and_preprocess_records(os.path.join(train_dir, "train_source2.tsv"), max_rows=args.max_cand_sample)
    s3_train = load_and_preprocess_records(os.path.join(train_dir, "train_source3.tsv"), max_rows=args.max_cand_sample)
    gt_map = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    
    cand_train_df = pl.concat([s2_train, s3_train])
    train_blocker = CandidateBlocker(max_candidates_per_entity=30)
    train_blocker.index_candidates(
        cand_train_df['entity_id'].to_list(),
        cand_train_df['country'].to_list(),
        cand_train_df['stripped_name'].to_list(),
        cand_train_df['pin'].to_list(),
        cand_train_df['metaphone'].to_list()
    )
    
    cand_dict_train = {row['entity_id']: row for row in cand_train_df.iter_rows(named=True)}
    
    X_train = []
    y_train = []
    for s1 in tqdm(s1_train.iter_rows(named=True), total=len(s1_train), desc="Building Training Pairs"):
        s1_id = s1['entity_id']
        true_matches = gt_map.get(s1_id, set())
        candidates = train_blocker.find_candidates_for_entity(
            s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
        )
        for cid in true_matches:
            if cid in cand_dict_train:
                cand = cand_dict_train[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                    s1['pin'], s1['city'], s1['state'],
                    cid, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                    cand['pin'], cand['city'], cand['state']
                )
                X_train.append(feat)
                y_train.append(1)
        false_cands = [c for c in candidates if c not in true_matches and c in cand_dict_train]
        if false_cands:
            cand_neg_feats = []
            for cid in false_cands:
                cand = cand_dict_train[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                    s1['pin'], s1['city'], s1['state'],
                    cid, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                    cand['pin'], cand['city'], cand['state']
                )
                hard_score = feat[0] * 0.4 + feat[1] * 0.3 + feat[4] * 0.3
                cand_neg_feats.append((hard_score, feat))
            cand_neg_feats.sort(key=lambda x: x[0], reverse=True)
            for _, feat in cand_neg_feats[:4]:
                X_train.append(feat)
                y_train.append(0)
                
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Training LightGBM model on {len(X_train):,} pairs...")
    clf = EntityMatchClassifier(n_estimators=250, learning_rate=0.06)
    clf.fit(X_train, y_train)
    
    # Save the trained model to disk for future benchmarking / reuse
    import joblib
    model_path = os.path.join(args.out_dir, "entity_match_model.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    
    # Free memory
    del s1_train, s2_train, s3_train, cand_train_df, cand_dict_train, X_train, y_train
    
    # 2. Index Test Candidate Sources
    print("Loading and indexing test candidate sources (S2 + S3)...")
    s2_test = load_and_preprocess_records(os.path.join(test_dir, "test_source2.tsv"), max_rows=args.test_max_rows)
    s3_test = load_and_preprocess_records(os.path.join(test_dir, "test_source3.tsv"), max_rows=args.test_max_rows)
    cand_test_df = pl.concat([s2_test, s3_test])
    del s2_test, s3_test
    
    test_blocker = CandidateBlocker(max_candidates_per_entity=30)
    test_blocker.index_candidates(
        cand_test_df['entity_id'].to_list(),
        cand_test_df['country'].to_list(),
        cand_test_df['stripped_name'].to_list(),
        cand_test_df['pin'].to_list(),
        cand_test_df['metaphone'].to_list()
    )
    cand_dict_test = {row['entity_id']: row for row in cand_test_df.iter_rows(named=True)}
    
    # 3. Process Test Source 1 Entities
    print("Loading and running inference on test_source1.tsv...")
    s1_test = load_and_preprocess_records(os.path.join(test_dir, "test_source1.tsv"), max_rows=args.test_max_rows)
    
    test_candidates_dict: Dict[str, List[str]] = {}
    test_matches_dict: Dict[str, List[str]] = {}
    
    threshold = args.threshold
    print(f"Applying decision threshold = {threshold:.2f}")
    
    batch_s1_ids = []
    batch_cids = []
    batch_features = []
    
    for s1 in tqdm(s1_test.iter_rows(named=True), total=len(s1_test), desc="Test Inference"):
        s1_id = s1['entity_id']
        cands = test_blocker.find_candidates_for_entity(
            s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
        )
        test_candidates_dict[s1_id] = cands
        test_matches_dict[s1_id] = []
        
        for cid in cands:
            if cid in cand_dict_test:
                cand = cand_dict_test[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                    s1['pin'], s1['city'], s1['state'],
                    cid, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                    cand['pin'], cand['city'], cand['state']
                )
                batch_s1_ids.append(s1_id)
                batch_cids.append(cid)
                batch_features.append(feat)
                
        # Batch inference every 50,000 pairs to conserve memory
        if len(batch_features) >= 50000:
            probs = clf.predict_proba(np.array(batch_features))
            for sid, cid, p in zip(batch_s1_ids, batch_cids, probs):
                if p >= threshold:
                    test_matches_dict[sid].append(cid)
            batch_s1_ids = []
            batch_cids = []
            batch_features = []
            
    # Process remaining pairs
    if batch_features:
        probs = clf.predict_proba(np.array(batch_features))
        for sid, cid, p in zip(batch_s1_ids, batch_cids, probs):
            if p >= threshold:
                test_matches_dict[sid].append(cid)
                
    # 4. Save Outputs in exact test_source1.tsv row order
    cand_file = os.path.join(args.out_dir, "candidate_pairs.tsv")
    match_file = os.path.join(args.out_dir, "matching_results.tsv")
    ordered_s1_ids = s1_test['entity_id'].to_list()
    write_candidate_pairs(test_candidates_dict, cand_file, ordered_s1_ids=ordered_s1_ids)
    write_matching_results(test_matches_dict, match_file, ordered_s1_ids=ordered_s1_ids)
    print("Test pipeline completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["validate", "test"], default="validate")
    parser.add_argument("--data-dir", default="student_resource/dataset")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--train-sample", type=int, default=50000)
    parser.add_argument("--val-sample", type=int, default=10000)
    parser.add_argument("--max-cand-sample", type=int, default=200000)
    parser.add_argument("--test-max-rows", type=int, default=None, help="Limit test rows for test runs, None for full")
    parser.add_argument("--threshold", type=float, default=0.60, help="Classification probability cutoff")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "test":
        run_test_mode(args)
    else:
        # Validation mode logic
        print("=== RUNNING IN VALIDATION MODE ===")
        total_s1 = args.train_sample + args.val_sample
        s1_df = load_and_preprocess_records(os.path.join(args.data_dir, "train/train_source1.tsv"), max_rows=total_s1)
        s2_df = load_and_preprocess_records(os.path.join(args.data_dir, "train/train_source2.tsv"), max_rows=args.max_cand_sample)
        s3_df = load_and_preprocess_records(os.path.join(args.data_dir, "train/train_source3.tsv"), max_rows=args.max_cand_sample)
        
        # Load ground truth and filter by S1 IDs present in sample
        gt_raw = pl.read_csv(os.path.join(args.data_dir, "train/train_ground_truth.tsv"), separator='\t')
        s1_all_ids = set(s1_df['entity_id'].to_list())
        gt_filtered = gt_raw.filter(pl.col('source1_entity_id').is_in(list(s1_all_ids)))
        
        gt_map: Dict[str, Set[str]] = {}
        for s1, m_str in zip(gt_filtered['source1_entity_id'].to_list(), gt_filtered['matched_entity_ids'].to_list()):
            if m_str and isinstance(m_str, str) and m_str.strip():
                gt_map[s1] = set(x.strip() for x in m_str.split(',') if x.strip())
            else:
                gt_map[s1] = set()
        for s1 in s1_all_ids:
            if s1 not in gt_map:
                gt_map[s1] = set()
        
        s1_train = s1_df.slice(0, args.train_sample)
        s1_val = s1_df.slice(args.train_sample, args.val_sample)
        val_s1_ids = set(s1_val['entity_id'].to_list())
        val_gt = {k: gt_map[k] for k in val_s1_ids if k in gt_map}
        
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
        
        cand_dict = {row['entity_id']: row for row in cand_df.iter_rows(named=True)}
        
        X_train = []
        y_train = []
        train_rows = s1_train.iter_rows(named=True)
        for s1 in tqdm(train_rows, total=len(s1_train), desc="Dev Train Pairs"):
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
        
        print("Training LightGBM match classifier...")
        clf = EntityMatchClassifier(n_estimators=150, learning_rate=0.08)
        clf.fit(X_train, y_train)
        
        print("Running validation inference...")
        val_candidates_dict: Dict[str, List[str]] = {}
        val_pairs_s1: List[str] = []
        val_pairs_cid: List[str] = []
        val_pairs_feat: List[List[float]] = []
        
        val_rows = s1_val.iter_rows(named=True)
        for s1 in tqdm(val_rows, total=len(s1_val), desc="Validation Blocking"):
            s1_id = s1['entity_id']
            cands = blocker.find_candidates_for_entity(
                s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
            )
            val_candidates_dict[s1_id] = cands
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
                    
        val_probs = clf.predict_proba(np.array(val_pairs_feat)) if val_pairs_feat else np.array([])
        best_thresh, best_f05, history = optimize_threshold(
            val_pairs_s1, val_pairs_cid, val_probs, val_gt
        )
        print(f"\nOptimal Threshold: {best_thresh:.2f} with Macro F_0.5: {best_f05:.4f}")
        
        val_preds: Dict[str, List[str]] = {s1: [] for s1 in val_s1_ids}
        for s1, cid, p in zip(val_pairs_s1, val_pairs_cid, val_probs):
            if p >= best_thresh:
                val_preds[s1].append(cid)
                
        metrics = evaluate_predictions(val_preds, val_gt)
        print(f"Validation F_0.5: {metrics['macro_f05']:.4f} | Prec: {metrics['macro_precision']:.4f} | Recall: {metrics['macro_recall']:.4f}")
        
        cand_file = os.path.join(args.out_dir, "candidate_pairs.tsv")
        match_file = os.path.join(args.out_dir, "matching_results.tsv")
        write_candidate_pairs(val_candidates_dict, cand_file)
        write_matching_results(val_preds, match_file)

if __name__ == "__main__":
    main()
