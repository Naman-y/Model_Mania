"""
Blocking and Candidate Generation Module for Amazon ML Challenge 2026.
Generates candidate pairs (S1_i -> [S2_j, S3_k]) using a union of:
1. Exact PIN match
2. FAISS dense vector similarity on entity names (semantic retrieval)
3. Metaphone phonetic skeleton match
Restricted by canonicalized country.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
from tqdm import tqdm
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

STOPWORDS = {
    'the', 'and', 'of', 'in', 'for', 'on', 'at', 'to', 'a', 'an', 'is',
    'enterprise', 'enterprises', 'services', 'solutions', 'industries',
    'associates', 'group', 'india', 'international', 'global'
}

class CandidateBlocker:
    def __init__(self, max_candidates_per_entity: int = 30, use_faiss: bool = True):
        self.max_candidates = max_candidates_per_entity
        self.use_faiss = use_faiss and FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_faiss:
            # Load a lightweight, fast embedding model for business names
            print("Loading SentenceTransformer model for business name embeddings...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        else:
            self.embedder = None
            self.embedding_dim = None
        
        # Indices per country:
        # country -> pin -> list of entity_ids
        self.pin_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # country -> token -> list of entity_ids
        self.token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # country -> metaphone_key -> list of entity_ids
        self.phonetic_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # FAISS indices per country
        # country -> {'index': faiss_index, 'ids': [entity_id1, entity_id2, ...], 'embeddings': np.ndarray}
        self.faiss_indices: Dict[str, Dict] = {}
        
        self.entity_count = 0
        
    def index_candidates(
        self,
        candidate_ids: List[str],
        countries: List[str],
        stripped_names: List[str],
        pins: List[Optional[str]],
        metaphone_keys: List[str]
    ):
        """Index S2 and S3 records into multi-index lookup tables (PIN, Token, Phonetic, and FAISS)."""
        # Group candidates by country
        country_data: Dict[str, Dict] = defaultdict(lambda: {'names': [], 'ids': [], 'pins': [], 'metaphones': []})
        
        for c_id, country, s_name, pin, meta in zip(candidate_ids, countries, stripped_names, pins, metaphone_keys):
            country_data[country]['names'].append(s_name)
            country_data[country]['ids'].append(c_id)
            country_data[country]['pins'].append(pin)
            country_data[country]['metaphones'].append(meta)
            self.entity_count += 1
            
            # 1. PIN index
            if pin:
                self.pin_index[country][pin].append(c_id)
                if len(pin) >= 3:
                    self.pin_index[country][pin[:3]].append(c_id)
            
            # 2. Token inverted index
            tokens = [t for t in s_name.split() if len(t) >= 3 and t not in STOPWORDS]
            for tok in tokens:
                self.token_index[country][tok].append(c_id)
                
            # 3. Phonetic index
            if meta:
                first_meta = meta.split()[0]
                if len(first_meta) >= 3:
                    self.phonetic_index[country][first_meta].append(c_id)
        
        # Build FAISS index per country if enabled
        if self.use_faiss:
            print(f"Building dense embeddings and FAISS indices for {len(country_data)} countries...")
            for country in tqdm(country_data, desc="Indexing FAISS by country"):
                data = country_data[country]
                if not data['names']:
                    continue
                # Encode all names for this country
                embeddings = self.embedder.encode(data['names'], batch_size=64, convert_to_numpy=True, show_progress_bar=False)
                embeddings = embeddings.astype('float32')
                
                # Create FAISS inner product index
                index = faiss.IndexFlatIP(self.embedding_dim)
                faiss.normalize_L2(embeddings)
                index.add(embeddings)
                
                self.faiss_indices[country] = {
                    'index': index,
                    'ids': data['ids']
                }
                            
    def find_candidates_for_entity(
        self,
        country: str,
        stripped_name: str,
        pin: Optional[str],
        metaphone_key: str
    ) -> List[str]:
        """
        Generate ranked candidate list for a single S1 entity using hybrid retrieval:
        1. Exact/Prefix PIN match (+3.0)
        2. Token inverted index match (+2.0)
        3. FAISS dense semantic similarity (+2.0 * sim)
        4. Phonetic metaphone match (+0.5)
        """
        candidate_scores: Dict[str, float] = defaultdict(float)
        
        # Pass 1: PIN matching (high weight)
        if pin and pin in self.pin_index[country]:
            for c_id in self.pin_index[country][pin][:20]:
                candidate_scores[c_id] += 3.0
        elif pin and len(pin) >= 3 and pin[:3] in self.pin_index[country]:
            for c_id in self.pin_index[country][pin[:3]][:10]:
                candidate_scores[c_id] += 1.0
        
        # Pass 2: Token inverted index (instant exact keyword match)
        tokens = [t for t in stripped_name.split() if len(t) >= 3 and t not in STOPWORDS]
        for tok in tokens:
            if tok in self.token_index[country]:
                matches = self.token_index[country][tok]
                if len(matches) < 5000:
                    for c_id in matches[:25]:
                        candidate_scores[c_id] += 2.0
                        
        # Pass 3: FAISS semantic similarity (dense retrieval)
        if self.use_faiss and country in self.faiss_indices:
            try:
                query_embedding = self.embedder.encode([stripped_name], batch_size=1, convert_to_numpy=True)[0]
                query_embedding = query_embedding.astype('float32')
                faiss.normalize_L2(query_embedding.reshape(1, -1))
                
                faiss_index = self.faiss_indices[country]['index']
                entity_ids = self.faiss_indices[country]['ids']
                
                distances, indices = faiss_index.search(query_embedding.reshape(1, -1), min(50, len(entity_ids)))
                for sim_score, idx in zip(distances[0], indices[0]):
                    if 0 <= idx < len(entity_ids):
                        c_id = entity_ids[idx]
                        candidate_scores[c_id] += float(sim_score) * 2.0
            except Exception as e:
                pass
        
        # Pass 4: Phonetic key (soundex/metaphone fallback)
        if metaphone_key:
            first_meta = metaphone_key.split()[0]
            if len(first_meta) >= 3 and first_meta in self.phonetic_index[country]:
                for c_id in self.phonetic_index[country][first_meta][:15]:
                    candidate_scores[c_id] += 0.5
        
        if not candidate_scores:
            return []
        
        # Sort candidates by score descending
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [c_id for c_id, score in sorted_candidates[:self.max_candidates]]
