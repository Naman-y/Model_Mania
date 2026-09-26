"""
Pairwise similarity feature engineering for business entity resolution.
Computes a fixed-width numeric feature vector for each (S1, Candidate) pair.
"""

from typing import Dict, Any, List, Optional
import jellyfish

def jaccard_similarity(tokens_a: set, tokens_b: set) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return float(intersection) / float(union) if union > 0 else 0.0

def compute_pairwise_features(
    s1_stripped_name: str,
    s1_metaphone: str,
    s1_clean_address: str,
    s1_pin: Optional[str],
    s1_city: Optional[str],
    s1_state: Optional[str],
    cand_id: str,
    cand_stripped_name: str,
    cand_metaphone: str,
    cand_clean_address: str,
    cand_pin: Optional[str],
    cand_city: Optional[str],
    cand_state: Optional[str]
) -> List[float]:
    """
    Computes a 9-dimensional numeric feature vector for an (S1, candidate) pair.
    """
    # 1. Name Levenshtein normalized similarity
    len_a = len(s1_stripped_name)
    len_b = len(cand_stripped_name)
    max_len = max(len_a, len_b, 1)
    lev_dist = jellyfish.levenshtein_distance(s1_stripped_name, cand_stripped_name)
    name_lev_sim = max(0.0, 1.0 - (lev_dist / max_len))
    
    # 2. Name token Jaccard
    s1_name_tokens = set(s1_stripped_name.split())
    cand_name_tokens = set(cand_stripped_name.split())
    name_jaccard = jaccard_similarity(s1_name_tokens, cand_name_tokens)
    
    # 3. Name length ratio
    name_len_ratio = min(len_a, len_b) / max_len
    
    # 4. Metaphone key similarity
    s1_meta_first = s1_metaphone.split()[0] if s1_metaphone else ""
    cand_meta_first = cand_metaphone.split()[0] if cand_metaphone else ""
    if s1_meta_first and cand_meta_first:
        if s1_meta_first == cand_meta_first:
            meta_sim = 1.0
        elif s1_meta_first[:3] == cand_meta_first[:3]:
            meta_sim = 0.5
        else:
            meta_sim = 0.0
    else:
        meta_sim = 0.0
        
    # 5. Address token Jaccard
    s1_addr_tokens = set(s1_clean_address.split())
    cand_addr_tokens = set(cand_clean_address.split())
    addr_jaccard = jaccard_similarity(s1_addr_tokens, cand_addr_tokens)
    
    # 6. PIN agreement
    # 1.0 = match, 0.5 = 3-digit prefix match, -1.0 = clash, 0.0 = missing
    if s1_pin and cand_pin:
        if s1_pin == cand_pin:
            pin_score = 1.0
        elif len(s1_pin) >= 3 and len(cand_pin) >= 3 and s1_pin[:3] == cand_pin[:3]:
            pin_score = 0.5
        else:
            pin_score = -1.0
    else:
        pin_score = 0.0
        
    # 7. City agreement
    if s1_city and cand_city:
        if s1_city == cand_city or s1_city in cand_city or cand_city in s1_city:
            city_score = 1.0
        else:
            city_score = -1.0
    else:
        city_score = 0.0
        
    # 8. State agreement
    if s1_state and cand_state:
        if s1_state == cand_state or s1_state in cand_state or cand_state in s1_state:
            state_score = 1.0
        else:
            state_score = -1.0
    else:
        state_score = 0.0
        
    # 9. Source indicator (S2 vs S3)
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    
    return [
        name_lev_sim,
        name_jaccard,
        name_len_ratio,
        meta_sim,
        addr_jaccard,
        pin_score,
        city_score,
        state_score,
        is_s2
    ]

FEATURE_NAMES = [
    'name_lev_sim',
    'name_jaccard',
    'name_len_ratio',
    'meta_sim',
    'addr_jaccard',
    'pin_score',
    'city_score',
    'state_score',
    'is_s2'
]
