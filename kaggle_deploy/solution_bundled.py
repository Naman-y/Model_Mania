import os, sys, subprocess
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "jellyfish", "sentence-transformers", "faiss-gpu", "polars", "lightgbm", "-q"], check=True)
except Exception:
    pass

# BUNDLED SOLUTION SCRIPT

# === preprocessor.py ===
import re
import jellyfish
from typing import Optional, Tuple, Set, List

try:
    from aksharamukha import transliterate
    HAS_AKSHARAMUKHA = True
except ImportError:
    HAS_AKSHARAMUKHA = False

# Legal suffixes to strip from business names for cleaner matching
LEGAL_SUFFIXES = {
    'corporation', 'corp', 'incorporated', 'inc', 'limited', 'ltd',
    'private limited', 'pvt ltd', 'pvt', 'private', 'llc', 'llp',
    'co', 'company', 'gmbh', 'sa', 'sarl', 'plc',
    'enterprise', 'enterprises'
}

# Regex patterns
RE_NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9\s]')
RE_STRIP_ALL = re.compile(r'[^a-zA-Z0-9]')
RE_MULTIPLE_SPACES = re.compile(r'\s+')
RE_AMPERSAND = re.compile(r'\s*&\s*')

# Postal code regex:
# 1) 6-digit Indian PIN (e.g. 560001, 560 001, 560-001) -> capture as 6 digits
RE_PIN_6DIGIT = re.compile(r'\b(\d{3})[\s-]?(\d{3})\b')
# 2) 5-digit US / France ZIP/Postal code (e.g. 90210, 75008)
RE_PIN_5DIGIT = re.compile(r'\b(\d{5})\b')

# Standard abbreviations for address normalization
ADDRESS_ABBR = {
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'blvd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'hwy': 'highway',
    'apt': 'apartment',
    'ste': 'suite',
    'bldg': 'building',
    'fl': 'floor',
    'opp': 'opposite',
    'nr': 'near'
}

def canonicalize_country(country_str: Optional[str]) -> str:
    """Canonicalize country strings to prevent silent blocking splits."""
    if not country_str or not isinstance(country_str, str):
        return "UNKNOWN"
    c = country_str.strip().lower()
    c = RE_STRIP_ALL.sub('', c)
    if c in {'us', 'usa', 'unitedstates', 'unitedstatesofamerica'}:
        return "US"
    if c in {'india', 'ind', 'in'}:
        return "India"
    if c in {'france', 'fr'}:
        return "France"
    return country_str.strip().title()

import re

# Fast detector: matches any char in actual non-Latin Unicode script blocks
# (Devanagari U+0900-U+097F, Bengali, Tamil, Telugu, Arabic, CJK, etc.)
# Does NOT trigger for Latin-with-diacritics (é, ñ, ü) — those are fine as-is
_RE_NON_LATIN_SCRIPT = re.compile(
    r'[\u0900-\u0D7F'   # Indic (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam)
    r'\u0E00-\u0E7F'    # Thai
    r'\u0600-\u06FF'    # Arabic
    r'\u0400-\u04FF'    # Cyrillic
    r'\u4E00-\u9FFF'    # CJK Unified Ideographs
    r'\uAC00-\uD7AF'    # Korean Hangul
    r'\u3040-\u30FF]'   # Japanese Hiragana/Katakana
)

def transliterate_to_latin(text: str) -> str:
    """Convert native-script text to Latin. Only calls aksharamukha when truly needed."""
    if not text:
        return ""
    # Fast O(1)-ish regex check — only call aksharamukha for real non-Latin scripts
    if not _RE_NON_LATIN_SCRIPT.search(text):
        return text   # ASCII or Latin-with-diacritics — skip transliteration entirely
    if HAS_AKSHARAMUKHA:
        try:
            return transliterate.process('autodetect', 'RomanReadable', text)
        except Exception:
            try:
                return transliterate.process('autodetect', 'ISO', text)
            except Exception:
                return text
    return text


# Web domain prefixes and suffixes
RE_WEB_PREFIX = re.compile(r'^(https?://)?(www\d?\.)?', re.IGNORECASE)
RE_TLD_SUFFIX = re.compile(r'\.(com|org|net|in|co|io|biz|info|us|gov|edu|ai|me|online|store|tech)(\.[a-z]{2})?$', re.IGNORECASE)
RE_CAMEL_CASE = re.compile(r'([a-z])([A-Z])')

def clean_business_name(name: Optional[str]) -> Tuple[str, str, str]:
    """
    Cleans business name:
    Returns (cleaned_name, stripped_name, metaphone_key).
    """
    if not name or not isinstance(name, str):
        return ("", "", "")
    
    # 1. Transliterate if non-ASCII
    text = transliterate_to_latin(name)
    
    # 2. Lowercase and replace & with and
    text = text.lower().strip()
    
    # Strip web prefixes and domain suffixes (e.g. www.sweetdeli.com -> sweetdeli)
    text = RE_WEB_PREFIX.sub('', text)
    text = RE_TLD_SUFFIX.sub('', text)
    text = RE_CAMEL_CASE.sub(r'\1 \2', text)
    
    text = RE_AMPERSAND.sub(' and ', text)
    
    # 3. Strip punctuation
    text = RE_NON_ALPHANUM.sub(' ', text)
    tokens = [t for t in RE_MULTIPLE_SPACES.sub(' ', text).strip().split() if t]
    cleaned_name = " ".join(tokens)
    
    # 4. Strip legal suffixes and web terms
    filtered_tokens = []
    # Check multi-word suffixes first
    joined = " " + cleaned_name + " "
    for suffix in ['private limited', 'pvt ltd']:
        if joined.endswith(f" {suffix} "):
            joined = joined[:-len(suffix)-2]
    
    tokens = joined.strip().split()
    WEB_STOPWORDS = {'com', 'org', 'net', 'www', 'http', 'https', 'co'}
    for tok in tokens:
        if tok not in LEGAL_SUFFIXES and tok not in WEB_STOPWORDS:
            filtered_tokens.append(tok)
            
    stripped_name = " ".join(filtered_tokens) if filtered_tokens else cleaned_name
    
    # 5. Metaphone phonetic skeleton
    # Jellyfish metaphone on the first 2-3 key tokens
    metaphone_tokens = [jellyfish.metaphone(tok) for tok in (filtered_tokens[:3] if filtered_tokens else tokens[:3])]
    metaphone_key = " ".join([m for m in metaphone_tokens if m])
    
    return cleaned_name, stripped_name, metaphone_key

def extract_pin(address: Optional[str], country: str = "UNKNOWN") -> Optional[str]:
    """
    Extract postal code using tolerant regex.
    Prioritizes 6-digit PIN for India, 5-digit for US/France.
    """
    if not address or not isinstance(address, str):
        return None
    
    c = country.upper()
    if c == "INDIA":
        # Check 6-digit first
        m = RE_PIN_6DIGIT.search(address)
        if m:
            return m.group(1) + m.group(2)
    elif c in {"US", "FRANCE"}:
        m = RE_PIN_5DIGIT.search(address)
        if m:
            return m.group(1)
            
    # Generic fallback
    m6 = RE_PIN_6DIGIT.search(address)
    if m6:
        return m6.group(1) + m6.group(2)
    m5 = RE_PIN_5DIGIT.search(address)
    if m5:
        return m5.group(1)
        
    return None

def extract_city_state(address: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Simple, bounded last-two-segments rule per Implementation Plan Section 4:
    Accept last 2 comma-separated segments as (city, state) ONLY if neither contains a digit
    and neither is a literal substring of the other. Else return (None, None).
    """
    if not address or not isinstance(address, str):
        return (None, None)
        
    parts = [p.strip().lower() for p in address.split(',') if p.strip()]
    if len(parts) < 2:
        return (None, None)
        
    candidate_city = RE_NON_ALPHANUM.sub(' ', parts[-2]).strip()
    candidate_state = RE_NON_ALPHANUM.sub(' ', parts[-1]).strip()
    
    # Rule 1: Neither contains digits
    if any(ch.isdigit() for ch in candidate_city) or any(ch.isdigit() for ch in candidate_state):
        return (None, None)
        
    # Rule 2: Non-empty after clean
    if not candidate_city or not candidate_state:
        return (None, None)
        
    # Rule 3: Reject only if identical or same token set (e.g. repeated state: "Tamil Nadu, Tamil Nadu")
    if candidate_city == candidate_state or set(candidate_city.split()) == set(candidate_state.split()):
        return (None, None)
        
    return (candidate_city, candidate_state)

def clean_address(address: Optional[str]) -> str:
    """Normalize full address string with abbreviation expansion."""
    if not address or not isinstance(address, str):
        return ""
    text = transliterate_to_latin(address).lower()
    text = RE_AMPERSAND.sub(' and ', text)
    text = RE_NON_ALPHANUM.sub(' ', text)
    tokens = text.split()
    expanded = [ADDRESS_ABBR.get(tok, tok) for tok in tokens]
    return " ".join(expanded)


# === blocking_production.py ===
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
        try:
            # GPU indices cannot be serialized directly
            if hasattr(faiss, 'index_gpu_to_cpu') and hasattr(index, 'd') and not hasattr(index, 'metric_type'): 
                # Very hacky check if it's a gpu index but faiss.index_gpu_to_cpu is the safest way
                index_to_save = faiss.index_gpu_to_cpu(index) if self.use_gpu else index
            else:
                index_to_save = faiss.index_gpu_to_cpu(index) if self.use_gpu else index
        except Exception:
            index_to_save = index # fallback
            
        faiss.write_index(index_to_save, path)
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


# === features.py ===
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

import re
RE_NUMBERS = re.compile(r'\b\d+\b')

def extract_primary_numbers(addr: str) -> List[str]:
    """Extract street/building/unit numbers from address string."""
    if not addr:
        return []
    return RE_NUMBERS.findall(addr)

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
    Computes a 12-dimensional numeric feature vector for an (S1, candidate) pair.
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
    
    # 4. Jaro-Winkler similarity
    name_jw = jellyfish.jaro_winkler_similarity(s1_stripped_name, cand_stripped_name)
    
    # 5. Exact stripped name match
    exact_name = 1.0 if s1_stripped_name == cand_stripped_name and s1_stripped_name else 0.0
    
    # 6. Metaphone key similarity
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
        
    # 7. Address token Jaccard
    s1_addr_tokens = set(s1_clean_address.split())
    cand_addr_tokens = set(cand_clean_address.split())
    addr_jaccard = jaccard_similarity(s1_addr_tokens, cand_addr_tokens)
    
    # 8. Building / Street number agreement (eliminates near-neighbor false positives)
    s1_nums = extract_primary_numbers(s1_clean_address)
    cand_nums = extract_primary_numbers(cand_clean_address)
    if s1_nums and cand_nums:
        if s1_nums[0] == cand_nums[0]:
            num_agreement = 1.0
        elif set(s1_nums) & set(cand_nums):
            num_agreement = 0.5
        else:
            num_agreement = -1.0
    else:
        num_agreement = 0.0
    
    # 9. PIN agreement
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
        
    # 10. City agreement
    if s1_city and cand_city:
        if s1_city == cand_city or s1_city in cand_city or cand_city in s1_city:
            city_score = 1.0
        else:
            city_score = -1.0
    else:
        city_score = 0.0
        
    # 11. State agreement
    if s1_state and cand_state:
        if s1_state == cand_state or s1_state in cand_state or cand_state in s1_state:
            state_score = 1.0
        else:
            state_score = -1.0
    else:
        state_score = 0.0
        
    # 12. Source indicator (S2 vs S3)
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    
    return [
        name_lev_sim,
        name_jaccard,
        name_len_ratio,
        name_jw,
        exact_name,
        meta_sim,
        addr_jaccard,
        num_agreement,
        pin_score,
        city_score,
        state_score,
        is_s2
    ]

FEATURE_NAMES = [
    'name_lev_sim',
    'name_jaccard',
    'name_len_ratio',
    'name_jw',
    'exact_name',
    'meta_sim',
    'addr_jaccard',
    'num_agreement',
    'pin_score',
    'city_score',
    'state_score',
    'is_s2'
]


# === model.py ===
"""
LightGBM Classifier for Business Entity Resolution.
Trains on pairwise similarity features with automatic class-imbalance weighting.
"""

import lightgbm as lgb
import numpy as np
from typing import Dict, Any, Optional

class EntityMatchClassifier:
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_state: int = 42
    ):
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'random_state': random_state,
            'verbose': -1,
            'n_jobs': -1
        }
        self.model: Optional[lgb.LGBMClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit LightGBM with computed scale_pos_weight = neg_count / pos_count
        to handle the extreme class imbalance per Implementation Plan Section 2.
        """
        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        
        scale_pos_weight = (neg_count / max(pos_count, 1))
        # Bound the scale_pos_weight slightly to prevent over-predicting false merges (which F0.5 punishes 2x)
        scale_pos_weight = min(scale_pos_weight, 15.0)
        
        self.model = lgb.LGBMClassifier(
            **self.params,
            scale_pos_weight=scale_pos_weight
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probability for candidate pairs."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]


# === evaluate.py ===
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


# === run_full_training_and_eval.py ===
#!/usr/bin/env python
"""
Full Pipeline Training & Evaluation for Kaggle.
Trains on 100% of the training data and evaluates on 100% of the training data.
Uses the optimized FAISS + Hybrid Blocker from `src/blocking.py`.
"""
import os
import sys
import gc
import time
import subprocess

# If on Kaggle, the src files will be uploaded alongside the script
if os.path.exists("/kaggle/working"):
    print("Running on Kaggle, using uploaded source files...")

import numpy as np
import polars as pl
import joblib
from tqdm import tqdm
import lightgbm as lgb

def find_dataset_dir() -> str:
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "train_source1.tsv" in files:
                return os.path.dirname(root)
    candidates = [
        "/kaggle/input/amazon-ml-challenge-2026/dataset",
        "/kaggle/input/amazon-ml-challenge-2026",
        "student_resource/dataset",
        "dataset",
        "../student_resource/dataset"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train", "train_source1.tsv")):
            return c
    raise FileNotFoundError("Could not locate dataset containing train/train_source1.tsv")

def find_src_dir() -> str:
    if os.path.exists("/kaggle/working/Model_Mania/code/business_entity_resolution/src"):
        return "/kaggle/working/Model_Mania/code/business_entity_resolution/src"
    candidates = [
        "code/business_entity_resolution/src",
        "../code/business_entity_resolution/src"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not locate src/")

SRC_DIR = find_src_dir()
sys.path.append(SRC_DIR)


import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def _preprocess_chunk(args):
    ids, names, addrs, countries = args
    c_names, s_names, metas, ctrs, pins, cities, states, c_addrs = [], [], [], [], [], [], [], []
    for name, addr, c in zip(names, addrs, countries):
        cc = canonicalize_country(c)
        cn, sn, meta = clean_business_name(name)
        pin = extract_pin(addr, cc)
        city, state = extract_city_state(addr)
        ca = clean_address(addr)
        c_names.append(cn); s_names.append(sn); metas.append(meta)
        ctrs.append(cc); pins.append(pin); cities.append(city)
        states.append(state); c_addrs.append(ca)
    return ids, ctrs, c_names, s_names, metas, pins, cities, states, c_addrs

def preprocess_df(df: pl.DataFrame, n_workers: int = None) -> pl.DataFrame:
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 4)
    ids = df['entity_id'].to_list()
    names = df['business_name'].to_list()
    addrs = df['business_address'].to_list()
    ctrys = df['country'].to_list()
    n = len(ids)

    if n < 50_000 or n_workers == 1:
        results = [_preprocess_chunk((ids, names, addrs, ctrys))]
    else:
        chunk_size = (n + n_workers - 1) // n_workers
        chunks = [(ids[i:i+chunk_size], names[i:i+chunk_size], addrs[i:i+chunk_size], ctrys[i:i+chunk_size]) for i in range(0, n, chunk_size)]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_preprocess_chunk, c): idx for idx, c in enumerate(chunks)}
            ordered = [None] * len(chunks)
            for f in as_completed(futures):
                ordered[futures[f]] = f.result()
            results = ordered

    all_ids, all_ctrs, all_cn, all_sn, all_meta = [], [], [], [], []
    all_pin, all_city, all_state, all_addr = [], [], [], []
    for r in results:
        all_ids.extend(r[0]); all_ctrs.extend(r[1]); all_cn.extend(r[2])
        all_sn.extend(r[3]);  all_meta.extend(r[4]); all_pin.extend(r[5])
        all_city.extend(r[6]); all_state.extend(r[7]); all_addr.extend(r[8])

    return pl.DataFrame({
        'entity_id': all_ids, 'country': all_ctrs, 'cleaned_name': all_cn,
        'stripped_name': all_sn, 'metaphone': all_meta, 'pin': all_pin,
        'city': all_city, 'state': all_state, 'clean_addr': all_addr
    })

def main():
    data_dir = find_dataset_dir()
    out_dir = "/kaggle/working/output" if os.path.exists("/kaggle/working") else "output"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {out_dir}")
    
    # 1. Load Ground Truth and Source 1
    print("\n--- 1. LOADING DATA ---")
    gt_raw = pl.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), separator='\t')
    s1_raw = pl.read_csv(os.path.join(data_dir, "train", "train_source1.tsv"), separator='\t')
    
    gt_map = {}
    for s1, m_str in zip(gt_raw['source1_entity_id'].to_list(), gt_raw['matched_entity_ids'].to_list()):
        if m_str and isinstance(m_str, str) and m_str.strip():
            gt_map[s1] = set(x.strip() for x in m_str.split(',') if x.strip())
        else:
            gt_map[s1] = set()
            
    print(f"Total S1 entities: {len(s1_raw):,}")
    s1_df = preprocess_df(s1_raw)
    del s1_raw
    
    s2_raw = pl.read_csv(os.path.join(data_dir, "train", "train_source2.tsv"), separator='\t')
    s3_raw = pl.read_csv(os.path.join(data_dir, "train", "train_source3.tsv"), separator='\t')
    cand_raw = pl.concat([s2_raw, s3_raw])
    del s2_raw, s3_raw
    
    print(f"Total candidate entities: {len(cand_raw):,}")
    cand_df = preprocess_df(cand_raw)
    del cand_raw
    
    # Store candidates as compact tuples for fast lookups
    cand_dict = {
        r['entity_id']: r for r in cand_df.iter_rows(named=True)
    }
    
    # 2. Block and Index Candidates
    print("\n--- 2. BUILDING CANDIDATE BLOCKER (FAISS + Hybrid) ---")
    blocker = ProductionCandidateBlocker(
        max_candidates_per_entity=30,
        device='cuda',
        use_gpu_faiss=True
    )
    blocker.index_candidates(
        cand_df['entity_id'].to_list(), cand_df['country'].to_list(),
        cand_df['stripped_name'].to_list(), cand_df['pin'].to_list(),
        cand_df['metaphone'].to_list()
    )
    del cand_df; gc.collect()
    
    # 3. Generate Training Pairs
    print("\n--- 3. GENERATING TRAINING PAIRS (100% of Data) ---")
    X_train, y_train = [], []
    
    # We will also keep track of candidates per S1 so we can reuse them for evaluation!
    s1_candidates_map = {}
    
    for s1 in tqdm(s1_df.iter_rows(named=True), total=len(s1_df), desc="Extracting Features"):
        s1_id = s1['entity_id']
        true_matches = gt_map.get(s1_id, set())
        
        cands = blocker.find_candidates_for_entity(
            s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone']
        )
        s1_candidates_map[s1_id] = cands
        
        # Add all positives
        for cid in true_matches:
            if cid in cand_dict:
                c = cand_dict[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'], s1['pin'], s1['city'], s1['state'],
                    cid, c['stripped_name'], c['metaphone'], c['clean_addr'], c['pin'], c['city'], c['state']
                )
                X_train.append(feat)
                y_train.append(1)
                
        # Add hard negatives
        false_cands = [c for c in cands if c not in true_matches and c in cand_dict]
        for cid in false_cands[:4]: # 1:4 Negative sampling
            c = cand_dict[cid]
            feat = compute_pairwise_features(
                s1['stripped_name'], s1['metaphone'], s1['clean_addr'], s1['pin'], s1['city'], s1['state'],
                cid, c['stripped_name'], c['metaphone'], c['clean_addr'], c['pin'], c['city'], c['state']
            )
            X_train.append(feat)
            y_train.append(0)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int8)
    print(f"Total pairs: {len(X_train):,} (Pos: {np.sum(y_train==1):,}, Neg: {np.sum(y_train==0):,})")
    
    # 4. Train LightGBM
    print("\n--- 4. TRAINING LIGHTGBM ---")
    clf = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', n_estimators=300,
        learning_rate=0.06, max_depth=7, num_leaves=63,
        scale_pos_weight=4.0, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    model_path = os.path.join(out_dir, "entity_match_model.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    
    del X_train, y_train; gc.collect()
    
    # 5. Evaluate on Entire Set
    print("\n--- 5. EVALUATING ON ENTIRE 100% TRAINING SET ---")
    val_s1_ids, val_cids, val_feat_list = [], [], []
    
    for s1 in tqdm(s1_df.iter_rows(named=True), total=len(s1_df), desc="Inference Prep"):
        s1_id = s1['entity_id']
        for cid in s1_candidates_map[s1_id]:
            if cid in cand_dict:
                c = cand_dict[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'], s1['pin'], s1['city'], s1['state'],
                    cid, c['stripped_name'], c['metaphone'], c['clean_addr'], c['pin'], c['city'], c['state']
                )
                val_s1_ids.append(s1_id)
                val_cids.append(cid)
                val_feat_list.append(feat)
                
    print("Running model predictions...")
    val_probs = clf.predict_proba(np.array(val_feat_list, dtype=np.float32))[:, 1]
    
    # Optimize threshold
    print("Optimizing threshold on 100% data...")
    best_thresh, best_f05, _ = optimize_threshold(val_s1_ids, val_cids, val_probs, gt_map)
    
    print(f"\nFinal Optimal Threshold: {best_thresh:.4f}")
    
    # Generate final matches
    final_matches = {sid: [] for sid in gt_map.keys()}
    for s1, cid, p in zip(val_s1_ids, val_cids, val_probs):
        if p >= best_thresh:
            final_matches[s1].append(cid)
            
    # Calculate full metrics
    final_metrics = evaluate_predictions(final_matches, gt_map)
    
    print(f"\n{'='*50}")
    print(f"FULL DATASET (100%) BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Macro F_0.5        : {final_metrics['macro_f05']:.4f}")
    print(f"Macro Precision    : {final_metrics['macro_precision']:.4f}")
    print(f"Macro Recall       : {final_metrics['macro_recall']:.4f}")
    
    # 6. Save Outputs for Error Analysis
    print("\n--- 6. SAVING PREDICTIONS ---")
    cand_path = os.path.join(out_dir, "candidate_pairs_train.tsv")
    match_path = os.path.join(out_dir, "matching_results_train.tsv")
    
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_df['entity_id'].to_list():
            f.write(f"{s1}\t{','.join(s1_candidates_map[s1])}\n")
            
    with open(match_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_df['entity_id'].to_list():
            f.write(f"{s1}\t{','.join(final_matches[s1])}\n")
            
    print("Done! Use these TSVs for comprehensive error analysis.")

if __name__ == "__main__":
    main()
