"""
Diagnostic Pipeline: Identifies where and why matches are missed at each stage.
Root cause analysis for entity resolution failures.

This script runs through the entire pipeline and tracks:
1. Which ground truth matches are MISSED by the blocker
2. Which blocker-retrieved matches are MISSED by the classifier
3. Which classifier predictions are MISSED by the threshold
4. Root cause for each missed match (spelling, abbreviation, low score, etc.)
"""

import os
import sys
import logging
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
from evaluate import evaluate_predictions

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DiagnosticAnalyzer:
    """Analyzes pipeline failures at each stage."""
    
    def __init__(self):
        self.diagnostics = {
            'blocker_missed': defaultdict(list),      # True matches not retrieved
            'classifier_missed': defaultdict(list),   # Retrieved but low score
            'threshold_missed': defaultdict(list),    # High score but below threshold
            'false_positives': defaultdict(list),     # Predicted but not true
            'missed_by_stage': {                       # Summary per stage
                'blocker': 0,
                'classifier': 0,
                'threshold': 0
            }
        }
    
    def analyze_blocker_failures(
        self,
        s1_id: str,
        true_matches: Set[str],
        blocker_candidates: List[str],
        s1_data: Dict,
        cand_dict: Dict,
        s2_s3_entities: Dict
    ):
        """Analyze why blocker failed to retrieve true matches."""
        missed_true = true_matches - set(blocker_candidates)
        
        for missed_id in missed_true:
            if missed_id not in cand_dict:
                continue
            
            missed_cand = cand_dict[missed_id]
            s1 = s1_data
            
            # Analyze why it was missed
            reason = self._diagnose_blocker_miss(
                s1, missed_cand,
                blocker_candidates, cand_dict
            )
            
            self.diagnostics['blocker_missed'][reason].append({
                's1_id': s1_id,
                'missed_id': missed_id,
                's1_name': s1['stripped_name'],
                'missed_name': missed_cand['stripped_name'],
                's1_country': s1['country'],
                'missed_country': missed_cand['country'],
                'reason': reason
            })
            self.diagnostics['missed_by_stage']['blocker'] += 1
    
    def analyze_classifier_failures(
        self,
        s1_id: str,
        true_matches: Set[str],
        blocker_candidates: List[str],
        classifier_scores: Dict[str, float],
        threshold: float
    ):
        """Analyze why classifier gave low scores to true matches."""
        # True matches that were retrieved but scored low
        retrieved_true = set(blocker_candidates) & true_matches
        
        for true_id in retrieved_true:
            score = classifier_scores.get(true_id, 0.0)
            if score < threshold:
                reason = self._diagnose_low_score(score)
                
                self.diagnostics['classifier_missed'][reason].append({
                    's1_id': s1_id,
                    'true_id': true_id,
                    'classifier_score': score,
                    'threshold': threshold,
                    'gap': threshold - score,
                    'reason': reason
                })
                self.diagnostics['missed_by_stage']['classifier'] += 1
    
    def analyze_threshold_failures(
        self,
        s1_id: str,
        true_matches: Set[str],
        blocker_candidates: List[str],
        classifier_scores: Dict[str, float],
        threshold: float
    ):
        """Analyze why threshold filtering removed true matches."""
        retrieved_true = set(blocker_candidates) & true_matches
        
        for true_id in retrieved_true:
            score = classifier_scores.get(true_id, 0.0)
            if score >= 0.5 and score < threshold:  # High enough but below threshold
                gap = threshold - score
                self.diagnostics['threshold_missed']['high_threshold'].append({
                    's1_id': s1_id,
                    'true_id': true_id,
                    'classifier_score': score,
                    'threshold': threshold,
                    'gap': gap
                })
                self.diagnostics['missed_by_stage']['threshold'] += 1
    
    def _diagnose_blocker_miss(
        self,
        s1: Dict,
        cand: Dict,
        retrieved: List[str],
        cand_dict: Dict
    ) -> str:
        """Determine why blocker missed this match."""
        # Check if it was close to being retrieved
        retrieved_names = [cand_dict[cid]['stripped_name'] for cid in retrieved[:30]]
        
        s1_name = s1['stripped_name'].lower()
        cand_name = cand['stripped_name'].lower()
        
        # Levenshtein distance
        from jellyfish import levenshtein_distance
        lev_dist = levenshtein_distance(s1_name, cand_name)
        
        if lev_dist / max(len(s1_name), len(cand_name)) > 0.7:
            return 'high_name_distance'
        
        # Token overlap
        s1_tokens = set(s1_name.split())
        cand_tokens = set(cand_name.split())
        overlap = len(s1_tokens & cand_tokens) / max(len(s1_tokens | cand_tokens), 1)
        
        if overlap < 0.3:
            return 'low_token_overlap'
        
        # PIN mismatch
        if s1['pin'] and cand['pin'] and s1['pin'] != cand['pin']:
            if len(s1['pin']) >= 3 and s1['pin'][:3] != cand['pin'][:3]:
                return 'pin_mismatch'
        
        # Country mismatch
        if s1['country'] != cand['country']:
            return 'country_mismatch'
        
        # Abbreviations/Spelling
        if any(t in s1_name for t in cand_name.split()) or \
           any(t in cand_name for t in s1_name.split()):
            return 'abbreviation_variation'
        
        # Transliteration issues
        if any(ord(c) > 127 for c in cand_name):
            return 'transliteration_mismatch'
        
        return 'unknown_reason'
    
    def _diagnose_low_score(self, score: float) -> str:
        """Classify why classifier gave low score."""
        if score < 0.2:
            return 'very_low_confidence'
        elif score < 0.4:
            return 'low_confidence'
        elif score < 0.6:
            return 'medium_confidence'
        elif score < 0.8:
            return 'high_confidence_but_below_threshold'
        return 'unknown'
    
    def generate_report(self) -> str:
        """Generate diagnostic report."""
        report = []
        report.append("=" * 80)
        report.append("DIAGNOSTIC PIPELINE REPORT: WHERE MATCHES ARE LOST")
        report.append("=" * 80)
        
        # Stage 1: Blocker failures
        report.append("\n" + "█" * 80)
        report.append("STAGE 1: BLOCKING (Blocker Recall)")
        report.append("█" * 80)
        report.append(f"\nTotal missed by blocker: {self.diagnostics['missed_by_stage']['blocker']}")
        
        blocker_reasons = defaultdict(int)
        for reason, items in self.diagnostics['blocker_missed'].items():
            blocker_reasons[reason] = len(items)
        
        report.append("\nRoot causes (why true matches were not retrieved):")
        for reason, count in sorted(blocker_reasons.items(), key=lambda x: -x[1]):
            pct = 100 * count / max(self.diagnostics['missed_by_stage']['blocker'], 1)
            report.append(f"  • {reason:.<40} {count:>5} ({pct:>5.1f}%)")
            
            # Show example
            if self.diagnostics['blocker_missed'][reason]:
                example = self.diagnostics['blocker_missed'][reason][0]
                report.append(f"      Example: {example['s1_name'][:30]:30} ≠ {example['missed_name'][:30]:30}")
        
        # Stage 2: Classifier failures
        report.append("\n" + "█" * 80)
        report.append("STAGE 2: CLASSIFICATION (Feature Scoring)")
        report.append("█" * 80)
        report.append(f"\nTotal missed by classifier: {self.diagnostics['missed_by_stage']['classifier']}")
        
        if self.diagnostics['classifier_missed']:
            avg_gap = np.mean([
                item['gap'] for items in self.diagnostics['classifier_missed'].values()
                for item in items
            ])
            report.append(f"Average score gap below threshold: {avg_gap:.4f}")
            
            report.append("\nConfidence distribution of missed true matches:")
            score_buckets = defaultdict(int)
            for items in self.diagnostics['classifier_missed'].values():
                for item in items:
                    score = item['classifier_score']
                    if score >= 0.8:
                        score_buckets['0.80-1.00'] += 1
                    elif score >= 0.6:
                        score_buckets['0.60-0.79'] += 1
                    elif score >= 0.4:
                        score_buckets['0.40-0.59'] += 1
                    else:
                        score_buckets['<0.40'] += 1
            
            for bucket in ['0.80-1.00', '0.60-0.79', '0.40-0.59', '<0.40']:
                count = score_buckets.get(bucket, 0)
                pct = 100 * count / max(self.diagnostics['missed_by_stage']['classifier'], 1)
                report.append(f"  • {bucket:.<40} {count:>5} ({pct:>5.1f}%)")
        
        # Stage 3: Threshold failures
        report.append("\n" + "█" * 80)
        report.append("STAGE 3: THRESHOLD (Decision Boundary)")
        report.append("█" * 80)
        report.append(f"\nTotal missed by threshold: {self.diagnostics['missed_by_stage']['threshold']}")
        
        if self.diagnostics['threshold_missed']['high_threshold']:
            gaps = [item['gap'] for item in self.diagnostics['threshold_missed']['high_threshold']]
            report.append(f"Average gap between score and threshold: {np.mean(gaps):.4f}")
            report.append(f"Could recover with threshold -0.05: {sum(1 for g in gaps if g < 0.05)}/{len(gaps)}")
        
        # Summary
        report.append("\n" + "=" * 80)
        report.append("SUMMARY: LEAKAGE AT EACH STAGE")
        report.append("=" * 80)
        
        total_lost = sum(self.diagnostics['missed_by_stage'].values())
        for stage, count in self.diagnostics['missed_by_stage'].items():
            pct = 100 * count / max(total_lost, 1)
            report.append(f"  {stage:.<30} {count:>5} matches ({pct:>5.1f}%)")
        
        report.append(f"\n  TOTAL LOST:                {total_lost:>5} matches")
        
        return "\n".join(report)


def run_diagnostic(
    data_dir: str = 'student_resource/dataset/train',
    num_s1: int = 500,
    num_cands: int = 10000,
    out_file: str = 'diagnostic_report.txt'
):
    """Run full diagnostic pipeline."""
    
    print("DIAGNOSTIC PIPELINE: Root Cause Analysis")
    print("=" * 80)
    
    # Load data
    print(f"\n1️⃣  Loading {num_s1:,} S1 entities...")
    s1_raw = pl.read_csv(
        os.path.join(data_dir, 'train_source1.tsv'),
        separator='\t',
        n_rows=num_s1
    )
    
    print(f"2️⃣  Loading {num_cands:,} candidate entities...")
    s2_raw = pl.read_csv(
        os.path.join(data_dir, 'train_source2.tsv'),
        separator='\t',
        n_rows=num_cands // 2
    )
    s3_raw = pl.read_csv(
        os.path.join(data_dir, 'train_source3.tsv'),
        separator='\t',
        n_rows=num_cands // 2
    )
    
    print(f"3️⃣  Loading ground truth...")
    gt_raw = pl.read_csv(os.path.join(data_dir, 'train_ground_truth.tsv'), separator='\t')
    
    # Preprocess
    print(f"4️⃣  Preprocessing...")
    
    def preprocess_df(df):
        entity_ids = df['entity_id'].to_list()
        cleaned_names = []
        stripped_names = []
        metaphones = []
        countries = []
        pins = []
        cities = []
        states = []
        clean_addrs = []
        
        for name, addr, c in zip(
            df['business_name'].to_list(),
            df['business_address'].to_list(),
            df['country'].to_list()
        ):
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
            'city': city,
            'state': states,
            'clean_addr': clean_addrs
        })
    
    s1_df = preprocess_df(s1_raw)
    cand_df = pl.concat([preprocess_df(s2_raw), preprocess_df(s3_raw)])
    
    # Load ground truth
    s1_ids = set(s1_df['entity_id'].to_list())
    gt_filtered = gt_raw.filter(pl.col('source1_entity_id').is_in(list(s1_ids)))
    
    gt_map = {}
    for s1, matched_str in zip(
        gt_filtered['source1_entity_id'].to_list(),
        gt_filtered['matched_entity_ids'].to_list()
    ):
        if matched_str and isinstance(matched_str, str) and matched_str.strip():
            gt_map[s1] = set(m.strip() for m in matched_str.split(',') if m.strip())
        else:
            gt_map[s1] = set()
    
    # Build blocker
    print(f"5️⃣  Building FAISS blocker...")
    blocker = ProductionCandidateBlocker(max_candidates_per_entity=30)
    blocker.index_candidates(
        cand_df['entity_id'].to_list(),
        cand_df['country'].to_list(),
        cand_df['stripped_name'].to_list(),
        cand_df['pin'].to_list(),
        cand_df['metaphone'].to_list(),
        persist_indices=False
    )
    
    cand_dict = {row['entity_id']: row for row in cand_df.iter_rows(named=True)}
    
    # Run diagnostic
    print(f"6️⃣  Running diagnostic analysis...")
    analyzer = DiagnosticAnalyzer()
    
    classifier_scores = defaultdict(dict)
    
    for s1 in tqdm(s1_df.iter_rows(named=True), total=len(s1_df), desc="Analyzing"):
        s1_id = s1['entity_id']
        if s1_id not in gt_map:
            continue
        
        true_matches = gt_map[s1_id]
        if not true_matches:
            continue
        
        # Get blocker candidates
        candidates = blocker.find_candidates_for_entity(
            s1['country'],
            s1['stripped_name'],
            s1['pin'],
            s1['metaphone']
        )
        
        # Analyze blocker failures
        analyzer.analyze_blocker_failures(
            s1_id, true_matches, candidates,
            s1, cand_dict, None
        )
        
        # Compute features and classifier scores
        for cand_id in candidates:
            if cand_id not in cand_dict:
                continue
            
            cand = cand_dict[cand_id]
            try:
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'],
                    s1['pin'], s1['city'], s1['state'],
                    cand_id, cand['stripped_name'], cand['metaphone'], cand['clean_addr'],
                    cand['pin'], cand['city'], cand['state']
                )
                classifier_scores[s1_id][cand_id] = feat  # Will need classifier to score
            except:
                pass
        
        # Analyze classifier failures (using simple heuristic for now)
        threshold = 0.85
        # TODO: Need trained classifier for real analysis
    
    # Generate report
    report = analyzer.generate_report()
    print("\n" + report)
    
    # Save report
    with open(out_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to {out_file}")
    
    return analyzer


if __name__ == "__main__":
    # Check if running from main project
    data_dir = 'student_resource/dataset/train'
    
    if not os.path.exists(data_dir):
        # Try from main checkout
        data_dir = os.path.join(
            os.path.dirname(__file__),
            '../../../../..', 'student_resource/dataset/train'
        )
    
    if os.path.exists(data_dir):
        analyzer = run_diagnostic(
            data_dir=data_dir,
            num_s1=2000,
            num_cands=50000,
            out_file='diagnostic_report.txt'
        )
    else:
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run from project root or provide correct path")
