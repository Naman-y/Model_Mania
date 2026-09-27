"""
Official metric: macro-averaged F_0.5 over Source-1 entities.
  - entity with no true matches: 1.0 if predicted empty else 0.0
  - otherwise F_0.5 = 1.25 P R / (0.25 P + R)   (0 when nothing correct)
"""

from typing import Dict, List, Set

import numpy as np
import polars as pl


def compute_entity_f05(pred: Set[str], truth: Set[str]) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def evaluate_predictions(predictions: Dict[str, List[str]], ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
    scores = [compute_entity_f05(set(predictions.get(s, [])), t) for s, t in ground_truth.items()]
    return {"macro_f05": float(np.mean(scores)) if scores else 0.0, "total_evaluated": len(scores)}


def macro_f05_frame(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> dict:
    """
    pred / truth: (s1, cid) pair frames; s1_ids: every Source-1 id being evaluated.
    Returns macro F0.5 plus diagnostics.
    """
    base = pl.DataFrame({"s1": s1_ids})
    n_pred = pred.group_by("s1").agg(pl.len().alias("n_pred"))
    n_true = truth.group_by("s1").agg(pl.len().alias("n_true"))
    tp = pred.join(truth, on=["s1", "cid"], how="inner").group_by("s1").agg(pl.len().alias("tp"))
    m = base.join(n_pred, on="s1", how="left").join(n_true, on="s1", how="left").join(tp, on="s1", how="left") \
            .fill_null(0)
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise(1.25 * p * r / (0.25 * p + r)))
    m = m.with_columns(f.alias("f05"))
    single = m.filter(pl.col("n_true") == 0)
    return {
        "macro_f05": m["f05"].mean(),
        "precision_pairs": tp["tp"].sum() / max(pred.height, 1),
        "recall_pairs": tp["tp"].sum() / max(truth.join(base, on="s1", how="semi").height, 1),
        "singleton_acc": single["f05"].mean() if single.height else None,
        "n_entities": m.height,
    }
