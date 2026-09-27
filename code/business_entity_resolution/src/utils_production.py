"""
Production utilities for business entity resolution pipeline.
Includes caching, monitoring, configuration management, and error handling.
"""

import os
import time
import json
import logging
import hashlib
from typing import Dict, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for production pipeline."""
    # Blocking
    max_candidates_per_entity: int = 30
    embedding_model: str = 'all-MiniLM-L6-v2'
    device: str = 'cpu'  # 'cpu' or 'cuda'
    use_gpu_faiss: bool = False
    faiss_search_k: int = 50
    
    # Preprocessing
    cache_cleaned_names: bool = True
    parallel_preprocessing: bool = True
    
    # Model
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 6
    scale_pos_weight_cap: float = 15.0
    
    # Evaluation
    threshold_sweep_granularity: float = 0.05  # Check every 0.05 on 0.0-1.0
    
    # Batch processing
    batch_size_pairs: int = 50000
    batch_size_inference: int = 50000
    
    # Caching
    cache_dir: str = '.cache'
    cache_embeddings: bool = True
    persist_faiss_indices: bool = True
    
    # Monitoring
    log_metrics_interval: int = 1000  # Log every N items
    enable_profiling: bool = False
    
    @classmethod
    def from_json(cls, json_path: str) -> 'PipelineConfig':
        """Load config from JSON file."""
        with open(json_path) as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, json_path: str):
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Saved config to {json_path}")


class FeatureCache:
    """Cache computed features to avoid recomputation."""
    
    def __init__(self, cache_dir: str = '.cache/features'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}
    
    def get_key(self, s1_id: str, cand_id: str) -> str:
        """Generate cache key."""
        pair = f"{s1_id}_{cand_id}"
        return hashlib.md5(pair.encode()).hexdigest()
    
    def get(self, s1_id: str, cand_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached feature."""
        key = self.get_key(s1_id, cand_id)
        if key in self.memory_cache:
            return self.memory_cache[key]
        return None
    
    def put(self, s1_id: str, cand_id: str, features: Dict[str, Any]):
        """Cache features."""
        key = self.get_key(s1_id, cand_id)
        self.memory_cache[key] = features
    
    def clear(self):
        """Clear cache."""
        self.memory_cache.clear()


class MetricsCollector:
    """Collect and aggregate metrics throughout pipeline execution."""
    
    def __init__(self):
        self.metrics = {
            'preprocessing_time': 0.0,
            'blocking_time': 0.0,
            'feature_computation_time': 0.0,
            'model_training_time': 0.0,
            'inference_time': 0.0,
            'pairs_generated': 0,
            'pairs_positive': 0,
            'pairs_negative': 0,
            'inference_pairs': 0,
            'model_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': []
        }
        self.start_times = {}
    
    def start_timer(self, key: str):
        """Start a timer for a phase."""
        self.start_times[key] = time.time()
    
    def end_timer(self, key: str):
        """End a timer and accumulate."""
        if key in self.start_times:
            elapsed = time.time() - self.start_times[key]
            if key in self.metrics:
                self.metrics[key] += elapsed
            del self.start_times[key]
            return elapsed
        return 0.0
    
    def record_metric(self, key: str, value: Any):
        """Record a metric."""
        if key in self.metrics and isinstance(self.metrics[key], (int, float)):
            if isinstance(value, (int, float)):
                self.metrics[key] += value
            else:
                self.metrics[key] = value
        else:
            self.metrics[key] = value
    
    def record_error(self, error: str):
        """Record an error."""
        self.metrics['errors'].append(error)
        logger.error(error)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            **self.metrics,
            'total_time': sum(
                v for k, v in self.metrics.items() 
                if isinstance(v, (int, float)) and k.endswith('_time')
            )
        }
    
    def log_summary(self):
        """Log metrics summary."""
        summary = self.get_summary()
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION METRICS")
        logger.info("=" * 60)
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"  {key:.<40} {value:.2f}")
            else:
                logger.info(f"  {key:.<40} {value}")


def timing_decorator(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - t_start
            logger.debug(f"{func.__name__} took {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - t_start
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    return wrapper


def cache_result(cache: FeatureCache) -> Callable:
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(s1_id: str, cand_id: str, *args, **kwargs):
            cached = cache.get(s1_id, cand_id)
            if cached is not None:
                return cached
            result = func(s1_id, cand_id, *args, **kwargs)
            cache.put(s1_id, cand_id, result)
            return result
        return wrapper
    return decorator


class HealthCheck:
    """Health checks for pipeline components."""
    
    @staticmethod
    def check_embedder(embedder) -> bool:
        """Verify embedder is working."""
        try:
            result = embedder.encode_batch(['test'], show_progress=False)
            return result.shape[0] == 1
        except Exception as e:
            logger.error(f"Embedder health check failed: {e}")
            return False
    
    @staticmethod
    def check_model(model) -> bool:
        """Verify model is working."""
        try:
            import numpy as np
            dummy_X = np.random.rand(10, 9)
            probs = model.predict_proba(dummy_X)
            return len(probs) == 10
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return False
    
    @staticmethod
    def check_faiss_index(index, n_test: int = 5) -> bool:
        """Verify FAISS index is working."""
        try:
            import faiss
            if index.ntotal == 0:
                return False
            dim = index.d
            query = np.random.rand(1, dim).astype('float32')
            distances, indices = index.search(query, min(5, index.ntotal))
            return len(indices[0]) > 0
        except Exception as e:
            logger.error(f"FAISS health check failed: {e}")
            return False


class RetryPolicy:
    """Retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retries."""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
        
        logger.error(f"All {self.max_retries} retries exhausted")
        raise last_exception


def setup_logging(log_file: Optional[str] = None, level: str = 'INFO'):
    """Configure logging for pipeline."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler if specified
    file_handler = None
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured at level {level}")
