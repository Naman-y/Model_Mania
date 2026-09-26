"""
Official macro-averaged F_0.5 evaluation metric and threshold optimizer.
Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
"""

from typing import Dict, List, Set, Tuple
import numpy as np

def compute_entity_f05(predicted_ids: Set[str], ground_truth_ids: Set[str]) -> float:
    """
    Computes F_0.5 for a single Source 1 entity.
    Singletons:
      - Correctly predicting empty -> 1.0
      - Predicting any match on a singleton -> 0.0
    Non-singletons:
      - Precision = |Pred intersect True| / |Pred|
      - Recall = |Pred intersect True| / |True|
      - F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if len(ground_truth_ids) == 0:
        return 1.0 if len(predicted_ids) == 0 else 0.0
        
    if len(predicted_ids) == 0:
        return 0.0
        
    tp = len(predicted_ids.intersection(ground_truth_ids))
    if tp == 0:
        return 0.0
        
    precision = tp / len(predicted_ids)
    recall = tp / len(ground_truth_ids)
    
    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0
        
    return (1.25 * precision * recall) / denominator

def evaluate_predictions(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Computes macro-averaged F_0.5, macro Precision, and macro Recall.
    Every entity in ground_truth must be evaluated.
    """
    f05_scores = []
    precisions = []
    recalls = []
    
    for s1_id, true_ids in ground_truth.items():
        pred_ids = set(predictions.get(s1_id, []))
        score = compute_entity_f05(pred_ids, true_ids)
        f05_scores.append(score)
        
        # Diagnostics
        if len(true_ids) == 0:
            if len(pred_ids) == 0:
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                precisions.append(0.0)
                recalls.append(1.0)
        else:
            tp = len(pred_ids.intersection(true_ids))
            precisions.append(tp / len(pred_ids) if pred_ids else 0.0)
            recalls.append(tp / len(true_ids))
            
    return {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "total_evaluated": len(ground_truth)
    }

def optimize_threshold(
    s1_ids: List[str],
    candidate_ids: List[str],
    scores: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    thresholds: List[float] = None
) -> Tuple[float, float, Dict[float, float]]:
    """
    Sweeps probability thresholds to find the threshold that maximizes macro F_0.5.
    Returns (best_threshold, best_f05, all_scores).
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.20, 0.90, 0.05)]
        
    # Group predictions by s1_id
    pair_scores: Dict[str, List[Tuple[str, float]]] = {s1: [] for s1 in ground_truth.keys()}
    for s1, cid, score in zip(s1_ids, candidate_ids, scores):
        if s1 in pair_scores:
            pair_scores[s1].append((cid, float(score)))
            
    best_thresh = 0.5
    best_f05 = -1.0
    history = {}
    
    for thresh in thresholds:
        preds = {}
        for s1, cands in pair_scores.items():
            matched = [cid for cid, s in cands if s >= thresh]
            preds[s1] = matched
            
        metrics = evaluate_predictions(preds, ground_truth)
        f05 = metrics["macro_f05"]
        history[thresh] = f05
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    return best_thresh, best_f05, history
