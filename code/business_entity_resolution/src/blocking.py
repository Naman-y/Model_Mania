"""
Blocking and Candidate Generation Module for Amazon ML Challenge 2026.
Generates candidate pairs (S1_i -> [S2_j, S3_k]) using a union of:
1. Exact PIN match
2. Token-overlap inverted index on stripped business names
3. Metaphone phonetic skeleton match
Restricted by canonicalized country.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
from tqdm import tqdm

STOPWORDS = {
    'the', 'and', 'of', 'in', 'for', 'on', 'at', 'to', 'a', 'an', 'is',
    'enterprise', 'enterprises', 'services', 'solutions', 'industries',
    'associates', 'group', 'india', 'international', 'global'
}

class CandidateBlocker:
    def __init__(self, max_candidates_per_entity: int = 30):
        self.max_candidates = max_candidates_per_entity
        # Inverted indices per country:
        # country -> token -> list of candidate entity_ids
        self.token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # country -> pin -> list of candidate entity_ids
        self.pin_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # country -> metaphone_key -> list of candidate entity_ids
        self.phonetic_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
    def index_candidates(
        self,
        candidate_ids: List[str],
        countries: List[str],
        stripped_names: List[str],
        pins: List[Optional[str]],
        metaphone_keys: List[str]
    ):
        """Index S2 and S3 records into memory-efficient lookup tables."""
        for c_id, country, s_name, pin, meta in zip(candidate_ids, countries, stripped_names, pins, metaphone_keys):
            # 1. PIN index
            if pin:
                self.pin_index[country][pin].append(c_id)
                # 3-digit prefix for partial matching
                if len(pin) >= 3:
                    self.pin_index[country][pin[:3]].append(c_id)
            
            # 2. Token inverted index
            tokens = [t for t in s_name.split() if len(t) >= 3 and t not in STOPWORDS]
            for tok in tokens:
                self.token_index[country][tok].append(c_id)
                
            # 3. Phonetic index
            if meta:
                # First metaphone code
                first_meta = meta.split()[0]
                if len(first_meta) >= 3:
                    self.phonetic_index[country][first_meta].append(c_id)
                    
    def find_candidates_for_entity(
        self,
        country: str,
        stripped_name: str,
        pin: Optional[str],
        metaphone_key: str
    ) -> List[str]:
        """
        Generate ranked candidate list for a single S1 entity.
        Returns a list of deduplicated candidate entity_ids.
        """
        candidate_scores: Dict[str, int] = defaultdict(int)
        
        # Pass 1: PIN matching (high weight)
        if pin and pin in self.pin_index[country]:
            for cid in self.pin_index[country][pin][:20]:
                candidate_scores[cid] += 3
        elif pin and len(pin) >= 3 and pin[:3] in self.pin_index[country]:
            for cid in self.pin_index[country][pin[:3]][:10]:
                candidate_scores[cid] += 1
                
        # Pass 2: Token overlap
        tokens = [t for t in stripped_name.split() if len(t) >= 3 and t not in STOPWORDS]
        for tok in tokens:
            if tok in self.token_index[country]:
                matches = self.token_index[country][tok]
                # If a token is too frequent (e.g. > 5000 records), skip or downweight
                if len(matches) < 5000:
                    for cid in matches[:25]:
                        candidate_scores[cid] += 2
                        
        # Pass 3: Phonetic key
        if metaphone_key:
            first_meta = metaphone_key.split()[0]
            if len(first_meta) >= 3 and first_meta in self.phonetic_index[country]:
                matches = self.phonetic_index[country][first_meta]
                if len(matches) < 2000:
                    for cid in matches[:15]:
                        candidate_scores[cid] += 1
                        
        if not candidate_scores:
            return []
            
        # Sort candidates by score descending
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, score in sorted_candidates[:self.max_candidates]]
