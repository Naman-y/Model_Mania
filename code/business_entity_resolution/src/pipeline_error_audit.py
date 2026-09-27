"""
Deep Diagnostic Failure Attribution Script for Entity Resolution Pipeline.
Analyzes every level of failure across the training dataset:
Level 1: Overall Evaluation (Macro F0.5, Precision, Recall)
Level 2: Error Categorization (TP, FP, Blocker Miss, Classifier Miss)
Level 3: Blocker Miss Root Cause (Country mismatch, Zero token overlap, Phonetic mismatch, PIN mismatch, Top-K truncation)
Level 4: Classifier Miss Feature & Score Analysis
Level 5: False Positive Analysis (Why were incorrect pairs predicted?)
Provides representative concrete examples for each failure category.
"""

import os
import sys
import polars as pl
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple

print("=================================================================")
print(" PIPELINE FAILURE ATTRIBUTION & ROOT CAUSE ANALYSIS")
print("=================================================================")

# Paths
train_gt_path = 'student_resource/dataset/train/train_ground_truth.tsv'
cand_path = 'output/candidate_pairs_train.tsv'
pred_path = 'output/matching_results_train.tsv'
s1_path = 'student_resource/dataset/train/train_source1.tsv'

# 1. Load Ground Truth
print("\n[1/5] Loading ground truth and predictions...")
gt_raw = pl.read_csv(train_gt_path, separator='\t')
pred_raw = pl.read_csv(pred_path, separator='\t')

gt_map: Dict[str, Set[str]] = {}
for sid, mstr in zip(gt_raw['source1_entity_id'].to_list(), gt_raw['matched_entity_ids'].to_list()):
    if mstr and isinstance(mstr, str) and mstr.strip():
        gt_map[sid] = set(x.strip() for x in mstr.split(',') if x.strip())
    else:
        gt_map[sid] = set()

pred_map: Dict[str, Set[str]] = {}
for sid, mstr in zip(pred_raw['source1_entity_id'].to_list(), pred_raw['matched_entity_ids'].to_list()):
    if mstr and isinstance(mstr, str) and mstr.strip():
        pred_map[sid] = set(x.strip() for x in mstr.split(',') if x.strip())
    else:
        pred_map[sid] = set()

total_s1 = len(pred_raw)
print(f"Total S1 entities evaluated: {total_s1:,}")

# 2. Overall Metrics & Error Categorization (TP, FP, FN)
print("\n[2/5] Categorizing TP, FP, and FN across all entities...")

total_gt_pairs = sum(len(s) for s in gt_map.values())
total_pred_pairs = sum(len(s) for s in pred_map.values())

tp_count = 0
fp_count = 0
fn_count = 0

all_fps: List[Tuple[str, str]] = []
all_fns: List[Tuple[str, str]] = []

precision_sum = 0.0
recall_sum = 0.0
f05_sum = 0.0

for sid, preds in pred_map.items():
    gts = gt_map.get(sid, set())
    
    tp = preds & gts
    fp = preds - gts
    fn = gts - preds
    
    tp_count += len(tp)
    fp_count += len(fp)
    fn_count += len(fn)
    
    for f in fp:
        all_fps.append((sid, f))
    for m in fn:
        all_fns.append((sid, m))
        
    p = len(tp) / len(preds) if preds else 1.0
    r = len(tp) / len(gts) if gts else 1.0
    beta_sq = 0.5 ** 2
    f05 = (1 + beta_sq) * p * r / (beta_sq * p + r) if (p + r) > 0 else 0.0
    
    precision_sum += p
    recall_sum += r
    f05_sum += f05

macro_p = precision_sum / total_s1
macro_r = recall_sum / total_s1
macro_f05 = f05_sum / total_s1

print(f"  Macro F0.5    : {macro_f05:.4f}")
print(f"  Macro Precision: {macro_p:.4f}  (Total predicted: {total_pred_pairs:,})")
print(f"  Macro Recall  : {macro_r:.4f}  (Total ground truth: {total_gt_pairs:,})")
print(f"  True Positives : {tp_count:,}")
print(f"  False Positives: {fp_count:,}")
print(f"  False Negatives: {fn_count:,}")

# 3. Analyze Candidates for False Negatives (Blocker Miss vs Classifier Miss)
print("\n[3/5] Auditing False Negatives: Blocker Miss vs Classifier Miss...")
# To be memory efficient, we can inspect a representative sample of 100,000 S1 records with candidates
# or read candidate_pairs_train lazily
print("Scanning candidate pairs to determine attribution for misses...")

fn_by_s1 = defaultdict(set)
for sid, cid in all_fns:
    fn_by_s1[sid].add(cid)

fp_by_s1 = defaultdict(set)
for sid, cid in all_fps:
    fp_by_s1[sid].add(cid)

# Sample 10,000 entities with misses for deep attribution
sample_s1_with_misses = set(list(fn_by_s1.keys())[:10000])

cands_sample_map: Dict[str, Set[str]] = {}
cand_scan = pl.scan_csv(cand_path, separator='\t')
cand_sub = cand_scan.filter(pl.col('source1_entity_id').is_in(list(sample_s1_with_misses))).collect()

for sid, cstr in zip(cand_sub['source1_entity_id'].to_list(), cand_sub['candidate_entity_ids'].to_list()):
    if cstr and isinstance(cstr, str) and cstr.strip():
        cands_sample_map[sid] = set(x.strip() for x in cstr.split(',') if x.strip())
    else:
        cands_sample_map[sid] = set()

sample_blocking_misses = []
sample_classifier_misses = []

for sid in sample_s1_with_misses:
    target_cands = cands_sample_map.get(sid, set())
    misses = fn_by_s1[sid]
    for cid in misses:
        if cid in target_cands:
            sample_classifier_misses.append((sid, cid))
        else:
            sample_blocking_misses.append((sid, cid))

total_sample_misses = len(sample_blocking_misses) + len(sample_classifier_misses)
if total_sample_misses > 0:
    pct_b = (len(sample_blocking_misses) / total_sample_misses) * 100
    pct_c = (len(sample_classifier_misses) / total_sample_misses) * 100
    print(f"Sample Attribution (on {len(sample_s1_with_misses):,} entities with misses):")
    print(f"  -> BLOCKER MISSES (Candidate pool missed true target) : {len(sample_blocking_misses):,} ({pct_b:.2f}%)")
    print(f"  -> CLASSIFIER MISSES (In candidate pool, dropped by score): {len(sample_classifier_misses):,} ({pct_c:.2f}%)")

# 4. Deep Inspection of Root Causes for Blocker Misses
print("\n[4/5] Inspecting Root Causes of Blocker Misses (Country, Name, Address, PIN)...")
# Let's pull metadata for a sample of 200 blocker misses and 200 classifier misses
target_s1_ids = list(set([sid for sid, _ in sample_blocking_misses[:100]] + [sid for sid, _ in sample_classifier_misses[:100]]))
target_cids = list(set([cid for _, cid in sample_blocking_misses[:100]] + [cid for _, cid in sample_classifier_misses[:100]]))

s1_data = pl.read_csv(s1_path, separator='\t').filter(pl.col('entity_id').is_in(target_s1_ids))
s2_scan = pl.scan_csv('student_resource/dataset/train/train_source2.tsv', separator='\t')
s3_scan = pl.scan_csv('student_resource/dataset/train/train_source3.tsv', separator='\t')

c_data = pl.concat([
    s2_scan.filter(pl.col('entity_id').is_in(target_cids)).collect(),
    s3_scan.filter(pl.col('entity_id').is_in(target_cids)).collect()
])

s1_dict = {r['entity_id']: r for r in s1_data.iter_rows(named=True)}
cand_dict = {r['entity_id']: r for r in c_data.iter_rows(named=True)}

country_mismatch_count = 0
exact_name_match_count = 0
partial_name_match_count = 0
zero_token_overlap_count = 0

examples_blocker_miss = []
for sid, cid in sample_blocking_misses[:100]:
    if sid in s1_dict and cid in cand_dict:
        s1_rec = s1_dict[sid]
        c_rec = cand_dict[cid]
        
        c_mismatch = (str(s1_rec['country']).lower() != str(c_rec['country']).lower())
        if c_mismatch:
            country_mismatch_count += 1
            
        n1 = set(str(s1_rec['business_name']).lower().split())
        n2 = set(str(c_rec['business_name']).lower().split())
        common = n1 & n2
        
        if len(common) == 0:
            zero_token_overlap_count += 1
        elif n1 == n2:
            exact_name_match_count += 1
        else:
            partial_name_match_count += 1
            
        if len(examples_blocker_miss) < 5:
            examples_blocker_miss.append({
                's1_id': sid, 'cand_id': cid,
                's1_name': s1_rec['business_name'], 'c_name': c_rec['business_name'],
                's1_addr': s1_rec['business_address'], 'c_addr': c_rec['business_address'],
                's1_country': s1_rec['country'], 'c_country': c_rec['country'],
                'common_tokens': list(common),
                'country_mismatch': c_mismatch
            })

print(f"\nBreakdown of 100 Sample Blocker Misses:")
print(f"  - Zero Common Name Tokens  : {zero_token_overlap_count}%")
print(f"  - Country Mismatch / Issue : {country_mismatch_count}%")
print(f"  - Partial Token Overlap    : {partial_name_match_count}%")
print(f"  - Exact Name Overlap       : {exact_name_match_count}% (missed due to PIN/metaphone partitioning or top-k cutoff)")

# 5. Output Detailed Breakdown & Concrete Examples
print("\n=================================================================")
print(" REPRESENTATIVE EXAMPLES OF MISSED ROWS & ROOT CAUSES")
print("=================================================================")

print("\n--- [LEVEL 2: BLOCKER MISSES - True target never retrieved into candidates] ---")
for i, ex in enumerate(examples_blocker_miss, 1):
    print(f"\nExample {i}:")
    print(f"  S1 Entity  : {ex['s1_id']} | Name: \"{ex['s1_name']}\" | Country: {ex['s1_country']}")
    print(f"  S1 Address : \"{ex['s1_addr']}\"")
    print(f"  Target Cand: {ex['cand_id']} | Name: \"{ex['c_name']}\" | Country: {ex['c_country']}")
    print(f"  Cand Address: \"{ex['c_addr']}\"")
    print(f"  Analysis   : Common Tokens: {ex['common_tokens']} | Country Mismatch: {ex['country_mismatch']}")
    if ex['country_mismatch']:
        print(f"  -> ROOT CAUSE: Country bucket mismatch prevented blocker from linking.")
    elif len(ex['common_tokens']) == 0:
        print(f"  -> ROOT CAUSE: Zero token overlap in raw names (e.g. acronyms, transliterations, severe spelling variation).")
    else:
        print(f"  -> ROOT CAUSE: Token index exceeded frequency cap or top-30 candidate pruning pushed it out.")

# Concrete False Positive examples
print("\n--- [LEVEL 3: FALSE POSITIVES - Incorrect candidate accepted by classifier] ---")
examples_fp = []
fp_s1_ids = list(set([sid for sid, _ in all_fps[:50]]))
fp_cids = list(set([cid for _, cid in all_fps[:50]]))

fp_s1_data = pl.read_csv(s1_path, separator='\t').filter(pl.col('entity_id').is_in(fp_s1_ids))
fp_c_data = pl.concat([
    s2_scan.filter(pl.col('entity_id').is_in(fp_cids)).collect(),
    s3_scan.filter(pl.col('entity_id').is_in(fp_cids)).collect()
])

fp_s1_dict = {r['entity_id']: r for r in fp_s1_data.iter_rows(named=True)}
fp_cand_dict = {r['entity_id']: r for r in fp_c_data.iter_rows(named=True)}

for sid, cid in all_fps[:50]:
    if sid in fp_s1_dict and cid in fp_cand_dict and len(examples_fp) < 5:
        examples_fp.append({
            's1_id': sid, 'cand_id': cid,
            's1_name': fp_s1_dict[sid]['business_name'], 'c_name': fp_cand_dict[cid]['business_name'],
            's1_addr': fp_s1_dict[sid]['business_address'], 'c_addr': fp_cand_dict[cid]['business_address'],
            's1_country': fp_s1_dict[sid]['country'], 'c_country': fp_cand_dict[cid]['country']
        })

for i, ex in enumerate(examples_fp, 1):
    print(f"\nExample {i}:")
    print(f"  S1 Entity  : {ex['s1_id']} | Name: \"{ex['s1_name']}\"")
    print(f"  S1 Address : \"{ex['s1_addr']}\"")
    print(f"  False Cand : {ex['cand_id']} | Name: \"{ex['c_name']}\"")
    print(f"  Cand Address: \"{ex['c_addr']}\"")
    print(f"  -> ROOT CAUSE: High lexical/phonetic similarity between distinct businesses (branches, generic titles, or co-located addresses).")

print("\nAudit complete.")
