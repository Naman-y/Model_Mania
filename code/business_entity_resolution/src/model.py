"""
Two-stage LightGBM matcher.

Stage 1: pairwise classifier on similarity + blocking features.
Stage 2: re-scores every pair using the stage-1 probabilities of the
         surrounding pairs - how the pair compares to the other candidates of
         the same Source-1 entity, and to the other Source-1 entities competing
         for the same Source-2/3 record (each S2/S3 record belongs to at most one
         Source-1 entity).
Decision: keep pairs with p2 >= threshold, and assign every S2/S3 record to at
most one Source-1 entity (the one with the highest p2).
"""

import lightgbm as lgb
import numpy as np
import polars as pl

from features import MODEL_FEATURES

STAGE2_EXTRA = ["p1", "p1_max", "p1_gap", "p1_rank", "p1_n50", "p1_sum",
                "pid_p1_other", "pid_p1_rank", "pid_n50"]
STAGE2_FEATURES = MODEL_FEATURES + STAGE2_EXTRA

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_estimators=2000, verbose=-1, n_jobs=-1)


def fit_lgb(X, y, X_val=None, y_val=None, **kw):
    params = dict(LGB_PARAMS, **kw)
    m = lgb.LGBMClassifier(**params)
    if X_val is not None:
        m.fit(X, y, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    else:
        m.fit(X, y)
    return m


def predict(models, X) -> np.ndarray:
    if not isinstance(models, (list, tuple)):
        models = [models]
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0).astype(np.float32)


def add_stage2_features(df: pl.DataFrame) -> pl.DataFrame:
    """df needs s1, cid, p1 (all pairs of the evaluation universe)."""
    p1 = pl.col("p1")
    df = df.with_columns(
        p1.max().over("s1").alias("p1_max"),
        (p1 - p1.max().over("s1")).alias("p1_gap"),
        p1.rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("p1_rank"),
        (p1 >= 0.5).sum().over("s1").cast(pl.Float32).alias("p1_n50"),
        p1.sum().over("s1").alias("p1_sum"),
        p1.rank("ordinal", descending=True).over("cid").cast(pl.Float32).alias("pid_p1_rank"),
        (p1 >= 0.5).sum().over("cid").cast(pl.Float32).alias("pid_n50"),
    )
    # best competing Source-1 entity for the same record (excluding this one)
    top2 = df.group_by("cid").agg(p1.top_k(2).alias("_t"))
    top2 = top2.with_columns(pl.col("_t").list.get(0).alias("_m1"),
                             pl.col("_t").list.get(1, null_on_oob=True).fill_null(0.0).alias("_m2")).drop("_t")
    df = df.join(top2, on="cid", how="left")
    df = df.with_columns(pl.when(p1 >= pl.col("_m1")).then(pl.col("_m2")).otherwise(pl.col("_m1"))
                         .alias("pid_p1_other")).drop("_m1", "_m2")
    return df


def decide(df: pl.DataFrame, threshold: float, score_col: str = "p2") -> pl.DataFrame:
    """Returns (s1, cid) matches: score >= threshold and best Source-1 entity for that record."""
    s = pl.col(score_col)
    d = df.filter(s >= threshold)
    d = d.filter(s.rank("ordinal", descending=True).over("cid") == 1)
    return d.select("s1", "cid")
