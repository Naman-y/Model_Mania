"""
Production-Grade FAISS Blocker for Amazon ML Challenge 2026.
Enterprise-ready entity resolution with:
- Model caching and GPU support
- FAISS index persistence and memory-mapping
- Batch embedding computation with progress tracking
- Comprehensive error handling and graceful degradation
- Configurable blocking strategies
- Monitoring and logging hooks
"""

import os
import pickle
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import polars as pl
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global model cache to avoid reloading
_EMBEDDER_CACHE = {}
_FAISS_INDEX_CACHE = {}

STOPWORDS = {
    'the', 'and', 'of', 'in', 'for', 'on', 'at', 'to', 'a', 'an', 'is',
    'enterprise', 'enterprises', 'services', 'solutions', 'industries',
    'associates', 'group', 'india', 'international', 'global'
}


class EmbeddingCache:
    """Persistent embedding cache for re-using computed vectors."""
    
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model."""
        import hashlib
        key = f"{model_name}_{hashlib.md5(text.encode()).hexdigest()}"
        return os.path.join(self.cache_dir, f"{key}.npy")
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if available."""
        try:
            cache_path = self.get_cache_key(text, model_name)
            if os.path.exists(cache_path):
                return np.load(cache_path)
        except Exception as e:
            logger.debug(f"Cache retrieval failed for {text}: {e}")
        return None
    
    def put(self, text: str, model_name: str, embedding: np.ndarray):
        """Store embedding in cache."""
        try:
            cache_path = self.get_cache_key(text, model_name)
            np.save(cache_path, embedding)
        except Exception as e:
            logger.debug(f"Cache storage failed for {text}: {e}")


class FastEmbedder:
    """Fast sentence embedding with GPU support, caching, and pooling."""
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = 'cpu',
        use_cache: bool = True,
        batch_size: int = 32
    ):
        global _EMBEDDER_CACHE
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache = EmbeddingCache() if use_cache else None
        
        # Use cached model if available
        cache_key = f"{model_name}_{device}"
        if cache_key in _EMBEDDER_CACHE:
            logger.info(f"Reusing cached embedder: {model_name}")
            self.model = _EMBEDDER_CACHE[cache_key]
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available")
            logger.info(f"Loading embedder: {model_name} on {device}")
            self.model = SentenceTransformer(model_name, device=device)
            _EMBEDDER_CACHE[cache_key] = self.model
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts in batches with progress bar and GPU acceleration."""
        embeddings = []
        iterator = tqdm(range(0, len(texts), self.batch_size), disable=not show_progress, desc="Embedding")
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            try:
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Encoding failed for batch {i}: {e}")
                # Fallback: zero embeddings for failed batch
                embeddings.append(np.zeros((len(batch), self.embedding_dim)))
        
        return np.vstack(embeddings) if embeddings else np.empty((0, self.embedding_dim))


class FAISSIndexManager:
    """Manages FAISS index creation, persistence, and querying."""
    
    def __init__(self, index_dir: str = ".cache/faiss_indices", use_gpu: bool = False):
        self.index_dir = index_dir
        self.use_gpu = use_gpu
        os.makedirs(index_dir, exist_ok=True)
    
    def build_index(self, embeddings: np.ndarray, metric: str = 'cosine') -> faiss.Index:
        """Build FAISS index with optional GPU acceleration."""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss not available")
        
        embeddings = embeddings.astype('float32')
        dim = embeddings.shape[1]
        
        if metric == 'cosine':
            # Normalize for cosine similarity (dot product on unit sphere)
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized vectors
        else:
            index = faiss.IndexFlatL2(dim)
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU")
            index = faiss.index_cpu_to_all_gpus(index)
        
        index.add(embeddings)
        logger.info(f"Built FAISS index: {index.ntotal} vectors, dimension {dim}")
        return index
    
    def save_index(self, index: faiss.Index, country: str):
        """Persist FAISS index to disk."""
        path = os.path.join(self.index_dir, f"{country}.faiss")
        faiss.write_index(index, path)
        logger.info(f"Saved FAISS index to {path}")
    
    def load_index(self, country: str) -> Optional[faiss.Index]:
        """Load FAISS index from disk."""
        path = os.path.join(self.index_dir, f"{country}.faiss")
        if os.path.exists(path):
            try:
                index = faiss.read_index(path)
                logger.info(f"Loaded FAISS index from {path}")
                return index
            except Exception as e:
                logger.error(f"Failed to load index {path}: {e}")
        return None


class ProductionCandidateBlocker:
    """
    Production-grade candidate blocker with:
    - Model caching
    - Index persistence
    - Batch processing
    - Error handling
    - Monitoring hooks
    """
    
    def __init__(
        self,
        max_candidates_per_entity: int = 30,
        embedding_model: str = 'all-MiniLM-L6-v2',
        device: str = 'cpu',
        use_gpu_faiss: bool = False,
        cache_dir: str = ".cache",
        enable_pin_fallback: bool = True,
        enable_phonetic_fallback: bool = True,
        faiss_search_k: int = 50
    ):
        self.max_candidates = max_candidates_per_entity
        self.embedding_model_name = embedding_model
        self.device = device
        self.cache_dir = cache_dir
        self.enable_pin_fallback = enable_pin_fallback
        self.enable_phonetic_fallback = enable_phonetic_fallback
        self.faiss_search_k = faiss_search_k
        
        # Initialize embedder and index manager
        try:
            self.embedder = FastEmbedder(
                model_name=embedding_model,
                device=device,
                use_cache=True,
                batch_size=32
            )
            logger.info(f"Initialized FastEmbedder with {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            self.embedder = None
        
        self.index_manager = FAISSIndexManager(
            index_dir=os.path.join(cache_dir, "faiss_indices"),
            use_gpu=use_gpu_faiss
        ) if FAISS_AVAILABLE else None
        
        # Country-specific indices
        self.pin_index: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
        self.phonetic_index: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
        self.faiss_indices: Dict[str, Dict[str, Any]] = {}
        
        # Metrics for monitoring
        self.metrics = {
            'total_candidates_indexed': 0,
            'countries': set(),
            'embedding_time': 0.0,
            'indexing_time': 0.0,
            'blocker_calls': 0,
            'fallback_count': 0
        }
    
    def index_candidates(
        self,
        candidate_ids: List[str],
        countries: List[str],
        stripped_names: List[str],
        pins: List[Optional[str]],
        metaphone_keys: List[str],
        persist_indices: bool = True
    ):
        """
        Index candidates with FAISS + PIN + phonetic.
        Supports persistence and batch processing.
        """
        import time
        
        logger.info(f"Indexing {len(candidate_ids):,} candidates across countries...")
        self.metrics['total_candidates_indexed'] = len(candidate_ids)
        
        # Group by country for efficient batch processing
        country_data: Dict[str, Dict[str, List]] = defaultdict(lambda: {
            'names': [], 'ids': [], 'pins': [], 'metaphones': []
        })
        
        for c_id, country, s_name, pin, meta in zip(
            candidate_ids, countries, stripped_names, pins, metaphone_keys
        ):
            country_data[country]['names'].append(s_name)
            country_data[country]['ids'].append(c_id)
            country_data[country]['pins'].append(pin)
            country_data[country]['metaphones'].append(meta)
            self.metrics['countries'].add(country)
        
        # Build indices per country
        for country in tqdm(country_data.keys(), desc="Indexing countries"):
            data = country_data[country]
            t_start = time.time()
            
            # Try FAISS first, fall back to PIN+phonetic on failure
            if self.embedder:
                try:
                    # Encode all names for this country
                    logger.info(f"  [{country}] Encoding {len(data['names']):,} names...")
                    embeddings = self.embedder.encode_batch(data['names'], show_progress=False)
                    embeddings = embeddings.astype('float32')
                    self.metrics['embedding_time'] += time.time() - t_start
                    
                    # Try to load existing index first (for incremental indexing)
                    index = self.index_manager.load_index(country) if self.index_manager else None
                    
                    if index is None:
                        # Build new FAISS index
                        index = self.index_manager.build_index(embeddings) if self.index_manager else None
                    
                    if index:
                        self.faiss_indices[country] = {
                            'index': index,
                            'ids': data['ids'],
                            'embeddings': embeddings
                        }
                        
                        if persist_indices:
                            self.index_manager.save_index(index, country)
                        
                        logger.info(f"  [{country}] FAISS index built: {index.ntotal} vectors")
                except Exception as e:
                    logger.error(f"  [{country}] FAISS indexing failed: {e}, using PIN+phonetic fallback")
                    self.metrics['fallback_count'] += 1
            
            # Always build PIN + phonetic indices (for fallback and re-ranking)
            for idx, (c_id, pin, meta) in enumerate(zip(
                data['ids'], data['pins'], data['metaphones']
            )):
                if pin:
                    self.pin_index[country][pin].append((c_id, idx))
                    if len(pin) >= 3:
                        self.pin_index[country][pin[:3]].append((c_id, idx))
                
                if meta:
                    first_meta = meta.split()[0]
                    if len(first_meta) >= 3:
                        self.phonetic_index[country][first_meta].append((c_id, idx))
            
            self.metrics['indexing_time'] += time.time() - t_start
    
    def find_candidates_for_entity(
        self,
        country: str,
        stripped_name: str,
        pin: Optional[str],
        metaphone_key: str,
        return_scores: bool = False
    ) -> List[str]:
        """
        Find candidate entities using multi-strategy blocking.
        Strategies tried in order:
        1. Exact PIN match (highest weight)
        2. FAISS semantic similarity (primary)
        3. Phonetic key (fallback)
        """
        self.metrics['blocker_calls'] += 1
        candidate_scores: Dict[str, float] = defaultdict(float)
        
        # Strategy 1: PIN matching (high weight)
        if self.enable_pin_fallback and pin:
            if pin in self.pin_index[country]:
                for c_id, idx in self.pin_index[country][pin][:20]:
                    candidate_scores[c_id] += 3.0
            elif len(pin) >= 3 and pin[:3] in self.pin_index[country]:
                for c_id, idx in self.pin_index[country][pin[:3]][:10]:
                    candidate_scores[c_id] += 1.0
        
        # Strategy 2: FAISS semantic similarity (primary blocker)
        if country in self.faiss_indices and self.embedder:
            try:
                query_embedding = self.embedder.model.encode(
                    [stripped_name], convert_to_numpy=True
                )[0].astype('float32')
                faiss.normalize_L2(query_embedding.reshape(1, -1))
                
                faiss_idx = self.faiss_indices[country]['index']
                entity_ids = self.faiss_indices[country]['ids']
                
                # Adaptive search k based on pool size
                search_k = min(self.faiss_search_k, len(entity_ids))
                distances, indices = faiss_idx.search(
                    query_embedding.reshape(1, -1), search_k
                )
                
                # Weight by similarity score (higher = better)
                for sim_score, idx in zip(distances[0], indices[0]):
                    if 0 <= idx < len(entity_ids):
                        c_id = entity_ids[idx]
                        candidate_scores[c_id] += float(sim_score) * 2.0
            except Exception as e:
                logger.warning(f"FAISS search failed for {country}: {e}, falling back to PIN+phonetic")
                self.metrics['fallback_count'] += 1
        
        # Strategy 3: Phonetic key (fallback)
        if self.enable_phonetic_fallback and metaphone_key:
            first_meta = metaphone_key.split()[0]
            if len(first_meta) >= 3 and first_meta in self.phonetic_index[country]:
                for c_id, idx in self.phonetic_index[country][first_meta][:15]:
                    candidate_scores[c_id] += 0.5
        
        if not candidate_scores:
            return []
        
        # Sort and return top candidates
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        results = [c_id for c_id, score in sorted_candidates[:self.max_candidates]]
        
        if return_scores:
            return results, {c_id: score for c_id, score in sorted_candidates[:self.max_candidates]}
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return blocking metrics for monitoring."""
        return {
            **self.metrics,
            'countries': list(self.metrics['countries']),
            'average_call_time_ms': (
                (self.metrics['embedding_time'] + self.metrics['indexing_time']) * 1000 / 
                max(self.metrics['blocker_calls'], 1)
            )
        }
    
    def log_metrics(self):
        """Log metrics to logger."""
        metrics = self.get_metrics()
        logger.info(f"Blocking Metrics:")
        logger.info(f"  Total indexed: {metrics['total_candidates_indexed']:,}")
        logger.info(f"  Countries: {len(metrics['countries'])}")
        logger.info(f"  Embedding time: {metrics['embedding_time']:.2f}s")
        logger.info(f"  Indexing time: {metrics['indexing_time']:.2f}s")
        logger.info(f"  Blocker calls: {metrics['blocker_calls']:,}")
        logger.info(f"  Fallback count: {metrics['fallback_count']}")
