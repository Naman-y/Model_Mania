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
        # country -> pin -> list of (entity_id, index_in_faiss)
        self.pin_index: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
        # country -> metaphone_key -> list of (entity_id, index_in_faiss)
        self.phonetic_index: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
        
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
        """Index S2 and S3 records using FAISS for dense retrieval."""
        if self.use_faiss:
            print("Building FAISS indices for candidate pool...")
        else:
            print("FAISS not available; falling back to PIN + phonetic matching only")
        
        # Group candidates by country
        country_data: Dict[str, Dict] = defaultdict(lambda: {'names': [], 'ids': [], 'pins': [], 'metaphones': []})
        
        for c_id, country, s_name, pin, meta in zip(candidate_ids, countries, stripped_names, pins, metaphone_keys):
            country_data[country]['names'].append(s_name)
            country_data[country]['ids'].append(c_id)
            country_data[country]['pins'].append(pin)
            country_data[country]['metaphones'].append(meta)
            self.entity_count += 1
        
        # Build FAISS index per country
        if self.use_faiss:
            print(f"Building embeddings and FAISS indices for {len(country_data)} countries...")
            for country in tqdm(country_data, desc="Indexing by country"):
                data = country_data[country]
                
                # Encode all names for this country
                embeddings = self.embedder.encode(data['names'], batch_size=32, convert_to_numpy=True)
                embeddings = embeddings.astype('float32')
                
                # Create FAISS index
                index = faiss.IndexFlatIP(self.embedding_dim)  # inner product (cosine similarity with normalized embeddings)
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                index.add(embeddings)
                
                self.faiss_indices[country] = {
                    'index': index,
                    'ids': data['ids'],
                    'embeddings': embeddings
                }
                
                # Index by PIN and phonetic
                for idx, (c_id, pin, meta) in enumerate(zip(data['ids'], data['pins'], data['metaphones'])):
                    if pin:
                        self.pin_index[country][pin].append((c_id, idx))
                        if len(pin) >= 3:
                            self.pin_index[country][pin[:3]].append((c_id, idx))
                    
                    if meta:
                        first_meta = meta.split()[0]
                        if len(first_meta) >= 3:
                            self.phonetic_index[country][first_meta].append((c_id, idx))
        else:
            # Fallback: only use PIN and phonetic indices
            for country in country_data:
                data = country_data[country]
                for idx, (c_id, pin, meta) in enumerate(zip(data['ids'], data['pins'], data['metaphones'])):
                    if pin:
                        self.pin_index[country][pin].append((c_id, idx))
                        if len(pin) >= 3:
                            self.pin_index[country][pin[:3]].append((c_id, idx))
                    if meta:
                        first_meta = meta.split()[0]
                        if len(first_meta) >= 3:
                            self.phonetic_index[country][first_meta].append((c_id, idx))
                            
    def find_candidates_for_entity(
        self,
        country: str,
        stripped_name: str,
        pin: Optional[str],
        metaphone_key: str
    ) -> List[str]:
        """
        Generate ranked candidate list for a single S1 entity.
        Uses FAISS semantic similarity as primary blocker, with PIN and phonetic as fallbacks.
        """
        candidate_scores: Dict[str, float] = defaultdict(float)
        
        # Pass 1: PIN matching (high weight)
        if pin and pin in self.pin_index[country]:
            for c_id, idx in self.pin_index[country][pin][:20]:
                candidate_scores[c_id] += 3.0
        elif pin and len(pin) >= 3 and pin[:3] in self.pin_index[country]:
            for c_id, idx in self.pin_index[country][pin[:3]][:10]:
                candidate_scores[c_id] += 1.0
        
        # Pass 2: FAISS semantic similarity (main blocker)
        if self.use_faiss and country in self.faiss_indices:
            try:
                # Encode the S1 query entity
                query_embedding = self.embedder.encode([stripped_name], batch_size=32, convert_to_numpy=True)[0]
                query_embedding = query_embedding.astype('float32')
                faiss.normalize_L2(query_embedding.reshape(1, -1))
                
                # Search FAISS index
                faiss_index = self.faiss_indices[country]['index']
                entity_ids = self.faiss_indices[country]['ids']
                
                # Search for top-50 similar entities (will rerank by score)
                distances, indices = faiss_index.search(query_embedding.reshape(1, -1), min(50, len(entity_ids)))
                
                # distances are similarity scores (higher = more similar)
                for sim_score, idx in zip(distances[0], indices[0]):
                    if idx >= 0 and idx < len(entity_ids):
                        c_id = entity_ids[idx]
                        # Convert similarity (0-2 range for normalized cosine) to score
                        candidate_scores[c_id] += float(sim_score) * 2.0
            except Exception as e:
                print(f"Warning: FAISS search failed for country {country}: {e}")
        
        # Pass 3: Phonetic key (fallback)
        if metaphone_key:
            first_meta = metaphone_key.split()[0]
            if len(first_meta) >= 3 and first_meta in self.phonetic_index[country]:
                for c_id, idx in self.phonetic_index[country][first_meta][:15]:
                    candidate_scores[c_id] += 0.5
        
        if not candidate_scores:
            return []
        
        # Sort candidates by score descending
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [c_id for c_id, score in sorted_candidates[:self.max_candidates]]
