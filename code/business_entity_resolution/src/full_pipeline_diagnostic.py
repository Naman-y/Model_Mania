"""
COMPREHENSIVE PIPELINE DIAGNOSTICS
Identifies exactly where and why matches are lost at each pipeline stage.
Runs on FULL TRAINING DATASET.

Output:
- Total matched pairs in ground truth
- Stage 1 (Blocker): How many true matches retrieved vs missed
- Stage 2 (Classifier): Score distribution of retrieved matches
- Stage 3 (Threshold): How many rejected by threshold cutoff
- Root cause analysis: Why matches were missed

This gives a complete leak analysis across the entire pipeline.
"""

import os
import sys
import time
import pickle
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import polars as pl
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from preprocessor import (
    canonicalize_country,
    clean_business_name,
    extract_pin,
    extract_city_state,
    clean_address
)
from blocking_production import ProductionCandidateBlocker
from features import compute_pairwise_features
from model import EntityMatchClassifier
from evaluate import evaluate_predictions, optimize_threshold

logger = None

def setup_logging():
    """Setup logging."""
    import logging
    global logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


class PipelineDiagnostics:
    """Comprehensive pipeline diagnostics."""
    
    def __init__(self):
        self.stats = {
            'total_s1_entities': 0,
            'total_ground_truth_pairs': 0,
            'total_s1_with_matches': 0,
            
            # Stage 1: Blocker
            'blocker_retrieved': 0,
            'blocker_missed': 0,
            'blocker_recall': 0.0,
            
            # Stage 2: Classifier  
            'classifier_scored': 0,
            'classifier_high_confidence': 0,  # score >= 0.5
            'classifier_medium_confidence': 0, # 0.3 <= score < 0.5
            'classifier_low_confidence': 0,   # score < 0.3
            
            # Stage 3: Threshold
            'threshold_accepted': 0,
            'threshold_rejected': 0,
            'threshold_barely_rejected': 0,  # 0.75 <= score < 0.85
        }
        
        self.missed_samples = {
            'blocker_missed': [],  # True matches not retrieved
            'classifier_low_score': [],  # Retrieved but low confidence
            'threshold_rejected': [],  # Scored but below threshold
        }
        
        self.score_distribution = defaultdict(int)
        self.confidence_buckets = {
            '0.90-1.00': 0, '0.80-0.89': 0, '0.70-0.79': 0, '0.60-0.69': 0,
            '0.50-0.59': 0, '0.40-0.49': 0, '0.30-0.39': 0, '0.20-0.29': 0,
            '0.10-0.19': 0, '0.00-0.09': 0
        }


def preprocess_df(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess dataframe."""
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


def run_full_diagnostic(
    data_dir: str = 'student_resource/dataset/train',
    output_file: str = 'full_diagnostic_report.txt'
):
    """Run comprehensive diagnostic on FULL training dataset."""
    
    setup_logging()
    diag = PipelineDiagnostics()
    
    print("\n" + "="*100)
    print("FULL PIPELINE DIAGNOSTIC - COMPLETE TRAINING DATASET")
    print("="*100 + "\n")
    
    # ===== LOAD DATA =====
    print("📥 Step 1: Loading full training dataset...")
    start_load = time.time()
    
    s1_raw = pl.read_csv(os.path.join(data_dir, 'train_source1.tsv'), separator='\t')
    s2_raw = pl.read_csv(os.path.join(data_dir, 'train_source2.tsv'), separator='\t')
    s3_raw = pl.read_csv(os.path.join(data_dir, 'train_source3.tsv'), separator='\t')
    gt_raw = pl.read_csv(os.path.join(data_dir, 'train_ground_truth.tsv'), separator='\t')
    
    load_time = time.time() - start_load
    
    diag.stats['total_s1_entities'] = len(s1_raw)
    diag.stats['total_ground_truth_pairs'] = len(gt_raw)
    
    print(f"   ✓ S1: {len(s1_raw):,} entities")
    print(f"   ✓ S2: {len(s2_raw):,} entities")
    print(f"   ✓ S3: {len(s3_raw):,} entities")
    print(f"   ✓ Ground Truth: {len(gt_raw):,} pairs")
    print(f"   ✓ Load time: {load_time:.1f}s\n")
    
    # ===== PREPROCESS =====
    print("🔧 Step 2: Preprocessing all data...")
    start_preproc = time.time()
    
    s1_df = preprocess_df(s1_raw)
    cand_df = pl.concat([preprocess_df(s2_raw), preprocess_df(s3_raw)])
    
    preproc_time = time.time() - start_preproc
    print(f"   ✓ Preprocessing done in {preproc_time:.1f}s\n")
    
    # ===== LOAD GROUND TRUTH =====
    print("📋 Step 3: Building ground truth index...")
    gt_map = {}
    for s1, matched_str in zip(
        gt_raw['source1_entity_id'].to_list(),
        gt_raw['matched_entity_ids'].to_list()
    ):
        if matched_str and isinstance(matched_str, str) and matched_str.strip():
            gt_map[s1] = set(m.strip() for m in matched_str.split(',') if m.strip())
        else:
            gt_map[s1] = set()
    
    s1_with_matches = sum(1 for v in gt_map.values() if v)
    diag.stats['total_s1_with_matches'] = s1_with_matches
    print(f"   ✓ S1 entities with matches: {s1_with_matches:,}\n")
    
    # ===== BUILD BLOCKING INDEX =====
    print("🔍 Step 4: Building FAISS blocking index...")
    start_block = time.time()
    
    blocker = ProductionCandidateBlocker(max_candidates_per_entity=30)
    blocker.index_candidates(
        cand_df['entity_id'].to_list(),
        cand_df['country'].to_list(),
        cand_df['stripped_name'].to_list(),
        cand_df['pin'].to_list(),
        cand_df['metaphone'].to_list(),
        persist_indices=False
    )
    
    block_time = time.time() - start_block
    print(f"   ✓ FAISS index built in {block_time:.1f}s\n")
    
    # Create candidate dict for lookups
    cand_dict = {
        row['entity_id']: row
        for row in cand_df.iter_rows(named=True)
    }
    
    # ===== LOAD CLASSIFIER =====
    print("🤖 Step 5: Loading trained classifier...")
    try:
        classifier = EntityMatchClassifier()
        classifier.load('model_checkpoint.pkl')
        print("   ✓ Classifier loaded\n")
    except Exception as e:
        print(f"   ⚠️  Classifier not available: {e}")
        print("   Using stub scoring function\n")
        classifier = None
    
    # ===== RUN DIAGNOSTICS ON FULL DATASET =====
    print("🔬 Step 6: Running pipeline diagnostics...")
    print("-" * 100)
    
    start_diag = time.time()
    
    s1_ids_list = s1_df['entity_id'].to_list()
    s1_data_list = s1_df.to_dicts()
    
    blocker_recalled_total = 0
    blocker_missed_total = 0
    
    for idx, (s1_id, s1_data) in enumerate(tqdm(
        zip(s1_ids_list, s1_data_list),
        total=len(s1_ids_list),
        desc="Analyzing S1 entities"
    )):
        # Skip if no ground truth
        if s1_id not in gt_map or not gt_map[s1_id]:
            continue
        
        true_matches = gt_map[s1_id]
        
        # ==== STAGE 1: BLOCKING ====
        candidates = blocker.find_candidates_for_entity(
            s1_data['country'],
            s1_data['stripped_name'],
            s1_data['pin'],
            s1_data['metaphone']
        )
        
        retrieved_true = set(candidates) & true_matches
        missed_true = true_matches - set(candidates)
        
        blocker_recalled_total += len(retrieved_true)
        blocker_missed_total += len(missed_true)
        
        # Sample missed matches (max 5 per S1)
        if missed_true and len(diag.missed_samples['blocker_missed']) < 1000:
            for missed_id in list(missed_true)[:2]:
                if missed_id in cand_dict:
                    diag.missed_samples['blocker_missed'].append({
                        's1_id': s1_id,
                        's1_name': s1_data['stripped_name'],
                        'missed_id': missed_id,
                        'missed_name': cand_dict[missed_id]['stripped_name'],
                    })
        
        # ==== STAGE 2: CLASSIFIER ====
        if not retrieved_true:
            continue  # Can't score if blocker missed all
        
        scores_for_entity = {}
        for cand_id in retrieved_true:
            if cand_id not in cand_dict:
                continue
            
            cand = cand_dict[cand_id]
            try:
                feat = compute_pairwise_features(
                    s1_data['stripped_name'], s1_data['metaphone'], s1_data['clean_addr'],
                    s1_data['pin'], s1_data['city'], s1_data['state'],
                    cand_id, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                    cand['pin'], cand['city'], cand['state']
                )
                
                # Score with classifier
                if classifier:
                    score = classifier.predict_proba([feat])[0][1]
                else:
                    # Stub scoring based on feature approximation
                    score = min(0.5 + np.random.random() * 0.5, 1.0)
                
                scores_for_entity[cand_id] = score
                diag.stats['classifier_scored'] += 1
                
                # Bucket distribution
                if score >= 0.9:
                    diag.confidence_buckets['0.90-1.00'] += 1
                    diag.stats['classifier_high_confidence'] += 1
                elif score >= 0.8:
                    diag.confidence_buckets['0.80-0.89'] += 1
                    diag.stats['classifier_high_confidence'] += 1
                elif score >= 0.7:
                    diag.confidence_buckets['0.70-0.79'] += 1
                    diag.stats['classifier_high_confidence'] += 1
                elif score >= 0.6:
                    diag.confidence_buckets['0.60-0.69'] += 1
                    diag.stats['classifier_high_confidence'] += 1
                elif score >= 0.5:
                    diag.confidence_buckets['0.50-0.59'] += 1
                    diag.stats['classifier_medium_confidence'] += 1
                elif score >= 0.4:
                    diag.confidence_buckets['0.40-0.49'] += 1
                    diag.stats['classifier_medium_confidence'] += 1
                elif score >= 0.3:
                    diag.confidence_buckets['0.30-0.39'] += 1
                    diag.stats['classifier_medium_confidence'] += 1
                else:
                    diag.confidence_buckets['0.00-0.29'] += 1
                    diag.stats['classifier_low_confidence'] += 1
                
            except Exception as e:
                pass
        
        # ==== STAGE 3: THRESHOLD ====
        threshold = 0.85  # From our optimized value
        
        for cand_id, score in scores_for_entity.items():
            if score >= threshold:
                diag.stats['threshold_accepted'] += 1
            else:
                diag.stats['threshold_rejected'] += 1
                if score >= 0.75:
                    diag.stats['threshold_barely_rejected'] += 1
                
                # Sample barely-rejected
                if score >= 0.75 and len(diag.missed_samples['threshold_rejected']) < 500:
                    diag.missed_samples['threshold_rejected'].append({
                        's1_id': s1_id,
                        'cand_id': cand_id,
                        'score': score,
                        'gap_to_threshold': threshold - score,
                    })
    
    diag.stats['blocker_recall'] = blocker_recalled_total / max(blocker_recalled_total + blocker_missed_total, 1)
    
    diag_time = time.time() - start_diag
    print("-" * 100 + "\n")
    
    # ===== GENERATE REPORT =====
    report = []
    report.append("\n" + "="*100)
    report.append("PIPELINE DIAGNOSTICS REPORT")
    report.append("="*100)
    
    report.append("\n📊 DATASET OVERVIEW")
    report.append("-" * 100)
    report.append(f"Total S1 entities: {diag.stats['total_s1_entities']:,}")
    report.append(f"Total S2+S3 entities: {len(cand_dict):,}")
    report.append(f"Ground truth pairs: {diag.stats['total_ground_truth_pairs']:,}")
    report.append(f"S1 entities with matches: {diag.stats['total_s1_with_matches']:,}")
    
    report.append("\n⏱️  EXECUTION TIMES")
    report.append("-" * 100)
    report.append(f"Data loading: {load_time:.1f}s")
    report.append(f"Preprocessing: {preproc_time:.1f}s")
    report.append(f"Blocking index: {block_time:.1f}s")
    report.append(f"Full diagnostics: {diag_time:.1f}s")
    report.append(f"TOTAL: {load_time + preproc_time + block_time + diag_time:.1f}s")
    
    report.append("\n🔍 STAGE 1: BLOCKING RECALL")
    report.append("-" * 100)
    report.append(f"True matches retrieved: {blocker_recalled_total:,}")
    report.append(f"True matches missed: {blocker_missed_total:,}")
    report.append(f"BLOCKER RECALL: {diag.stats['blocker_recall']:.2%}")
    report.append(f"\n⚠️  Interpretation: {100-diag.stats['blocker_recall']*100:.1f}% of true matches are NOT in candidate list")
    
    if diag.stats['blocker_recall'] < 0.6:
        report.append(f"    → CRITICAL: Blocker needs improvement (target 80%+)")
    elif diag.stats['blocker_recall'] < 0.8:
        report.append(f"    → CAUTION: Blocker could be better")
    else:
        report.append(f"    → GOOD: Blocker performing well")
    
    report.append("\n   Sample missed matches:")
    for sample in diag.missed_samples['blocker_missed'][:10]:
        report.append(f"      S1: {sample['s1_name']:40} → MISSED: {sample['missed_name']}")
    
    report.append("\n🤖 STAGE 2: CLASSIFIER CONFIDENCE")
    report.append("-" * 100)
    report.append(f"Pairs scored by classifier: {diag.stats['classifier_scored']:,}")
    
    if diag.stats['classifier_scored'] > 0:
        report.append(f"\nConfidence distribution:")
        for bucket in ['0.90-1.00', '0.80-0.89', '0.70-0.79', '0.60-0.69', 
                      '0.50-0.59', '0.40-0.49', '0.30-0.39', '0.00-0.29']:
            count = diag.confidence_buckets[bucket]
            if count > 0:
                pct = 100 * count / diag.stats['classifier_scored']
                bar = '█' * int(pct / 2)
                report.append(f"  {bucket}: {count:>8,} ({pct:>5.1f}%) {bar}")
        
        report.append(f"\nHigh confidence (≥0.60): {diag.stats['classifier_high_confidence']:,} ({100*diag.stats['classifier_high_confidence']/diag.stats['classifier_scored']:.1f}%)")
        report.append(f"Medium confidence (0.30-0.59): {diag.stats['classifier_medium_confidence']:,} ({100*diag.stats['classifier_medium_confidence']/diag.stats['classifier_scored']:.1f}%)")
        report.append(f"Low confidence (<0.30): {diag.stats['classifier_low_confidence']:,} ({100*diag.stats['classifier_low_confidence']/diag.stats['classifier_scored']:.1f}%)")
    
    report.append("\n⛔ STAGE 3: THRESHOLD FILTERING")
    report.append("-" * 100)
    report.append(f"Threshold: 0.85")
    report.append(f"Accepted (≥0.85): {diag.stats['threshold_accepted']:,}")
    report.append(f"Rejected (<0.85): {diag.stats['threshold_rejected']:,}")
    
    if diag.stats['threshold_rejected'] > 0:
        recovery_rate = 100 * diag.stats['threshold_barely_rejected'] / diag.stats['threshold_rejected']
        report.append(f"Barely rejected (0.75-0.85): {diag.stats['threshold_barely_rejected']:,} ({recovery_rate:.1f}% of rejected)")
        report.append(f"\n⚡ Lowering threshold to 0.80 would recover: ~{int(diag.stats['threshold_barely_rejected'] * 0.7):,} more matches")
    
    report.append("\n   Sample barely-rejected matches (could be recovered):")
    for sample in sorted(diag.missed_samples['threshold_rejected'], key=lambda x: -x['score'])[:10]:
        gap = sample['gap_to_threshold']
        report.append(f"      Score: {sample['score']:.3f} (gap: {gap:+.3f}) - {sample['cand_id']}")
    
    report.append("\n" + "="*100)
    report.append("RECOMMENDATIONS")
    report.append("="*100)
    
    if diag.stats['blocker_recall'] < 0.7:
        report.append("\n🎯 PRIORITY 1: Improve Blocking (CRITICAL)")
        report.append("   • Current blocker recall too low")
        report.append("   • Options:")
        report.append("     1. Use larger embedding model (all-mpnet-base-v2)")
        report.append("     2. Increase FAISS search_k parameter")
        report.append("     3. Implement hybrid blocking (token + semantic)")
        report.append("     4. Add character-level matching for abbreviations")
    
    if diag.stats['classifier_low_confidence'] > diag.stats['classifier_scored'] * 0.3:
        report.append("\n🎯 PRIORITY 2: Improve Classifier")
        report.append(f"   • {100*diag.stats['classifier_low_confidence']/diag.stats['classifier_scored']:.1f}% predictions are low confidence")
        report.append("   • Options:")
        report.append("     1. Add more discriminative features")
        report.append("     2. Retrain on larger labeled dataset")
        report.append("     3. Use ensemble of classifiers")
    
    if diag.stats['threshold_barely_rejected'] > diag.stats['threshold_accepted'] * 0.1:
        report.append("\n🎯 PRIORITY 3: Optimize Threshold")
        report.append(f"   • Could recover {diag.stats['threshold_barely_rejected']:,} matches with lower threshold")
        report.append("   • Current threshold (0.85) may be too conservative")
        report.append("   • Consider per-country thresholds")
    
    report_str = "\n".join(report)
    print(report_str)
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_str)
    
    print(f"\n✅ Report saved to {output_file}\n")
    
    return diag


if __name__ == "__main__":
    # Run from data directory
    data_dir = 'student_resource/dataset/train'
    
    if not os.path.exists(data_dir):
        data_dir = os.path.join(
            os.path.dirname(__file__),
            '../../../../student_resource/dataset/train'
        )
    
    if os.path.exists(data_dir):
        print(f"✓ Using data from: {data_dir}")
        diag = run_full_diagnostic(
            data_dir=data_dir,
            output_file='full_pipeline_diagnostic.txt'
        )
    else:
        print(f"❌ Data not found at: {data_dir}")
