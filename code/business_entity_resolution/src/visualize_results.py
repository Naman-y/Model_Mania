"""
Threshold curve report and ASCII visualizer for F_0.5 optimization.
"""

from typing import Dict, List, Tuple

def plot_ascii_f05_curve(history: Dict[float, float], optimal_thresh: float):
    """Prints a clean ASCII plot of the F_0.5 score across decision thresholds."""
    print("\n" + "=" * 55)
    print("      DECISION THRESHOLD vs MACRO F_0.5 CURVE")
    print("=" * 55)
    print(f" {'Threshold':<10} | {'Macro F_0.5':<12} | {'Bar Chart'}")
    print("-" * 55)
    
    max_val = max(history.values()) if history else 1.0
    min_val = min(history.values()) if history else 0.0
    val_range = max(max_val - min_val, 0.001)
    
    for t, score in sorted(history.items()):
        is_opt = " <-- OPTIMAL" if abs(t - optimal_thresh) < 1e-4 else ""
        norm_len = int(((score - min_val) / val_range) * 20) + 1
        bar = "█" * norm_len
        print(f" {t:<10.2f} | {score:<12.4f} | {bar}{is_opt}")
    print("=" * 55 + "\n")
