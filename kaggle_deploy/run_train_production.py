# type: ignore
import os, sys
try:
    import faiss
except ImportError:
    faiss = None

try:
    from evaluate import evaluate_predictions, optimize_threshold  # type: ignore
except ImportError:
    pass
