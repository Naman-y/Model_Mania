"""
Post-processing and Singleton Refinement for Amazon ML Challenge 2026.
Enforces all competition format and precision constraints:
1. Deduplication of matched IDs per entity.
2. Singleton thresholding: If max match confidence is below singleton_margin, emit empty list.
3. Candidate subset guarantee: Ensures matching_results is strictly a subset of candidate_pairs.
"""

from typing import Dict, List, Set

class SubmissionPostProcessor:
    def __init__(self, singleton_confidence_margin: float = 0.50):
        self.singleton_margin = singleton_confidence_margin
        
    def refine_predictions(
        self,
        raw_predictions: Dict[str, List[str]],
        candidate_map: Dict[str, List[str]],
        scores_map: Dict[str, Dict[str, float]] = None
    ) -> Dict[str, List[str]]:
        """
        Cleans and refines matches:
        - Deduplicates IDs while preserving ranking order
        - Ensures every matched ID is in candidates
        - Applies singleton thresholding
        """
        refined = {}
        for s1_id, match_list in raw_predictions.items():
            valid_cands = set(candidate_map.get(s1_id, []))
            
            seen = set()
            clean_matches = []
            for cid in match_list:
                # Must be a valid candidate and non-duplicate
                if cid in valid_cands and cid not in seen:
                    clean_matches.append(cid)
                    seen.add(cid)
                    
            # Singleton refinement if confidence scores are provided
            if scores_map and s1_id in scores_map:
                cand_scores = scores_map[s1_id]
                clean_matches = [
                    cid for cid in clean_matches
                    if cand_scores.get(cid, 0.0) >= self.singleton_margin
                ]
                
            refined[s1_id] = clean_matches
        return refined
