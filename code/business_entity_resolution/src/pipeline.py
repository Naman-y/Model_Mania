"""
End-to-end Business Entity Resolution pipeline.

  python pipeline.py all      --data-dir <dataset> --work-dir work --out-dir output
  python pipeline.py prepare  ...   normalise all source files -> parquet cache
  python pipeline.py pairs    ...   blocking + pair features   (--split train|test)
  python pipeline.py train    ...   two-stage LightGBM, validated on held-out Source-1 entities
  python pipeline.py predict  ...   test inference -> matching_results.tsv + candidate_pairs.tsv

Validation mirrors the test setting exactly: every train Source-1 record is
blocked against the FULL train Source-2/3 pool (~10M records), and 20% of the
Source-1 entities (by hash) are held out for scoring with the official metric.
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import polars as pl

from blocking import Blocker
from evaluate import macro_f05_frame
from features import MODEL_FEATURES, add_block_context, string_features
from model import STAGE2_FEATURES, add_stage2_features, decide, fit_lgb, predict
from prepare import load, load_ground_truth, prepare

N_FOLDS = 10
VAL_FOLDS = [0, 1]          # held-out Source-1 entities for scoring
TRAIN_FOLDS = [2, 3]        # stage-1 out-of-fold pair / stage-2 training
ES_FOLD = 4                 # early stopping


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def fold_expr():
    return (pl.col("s1").hash(seed=11) % N_FOLDS).cast(pl.Int32).alias("fold")


# --------------------------------------------------------------------------
# Blocking + features
# --------------------------------------------------------------------------
def build_pairs(work_dir, split, top_k, top_k_name, n_jobs, q_chunk=150000, q_batch=250000):
    out_dir = os.path.join(work_dir, f"pairs_{split}")
    os.makedirs(out_dir, exist_ok=True)
    s1 = load(work_dir, split, 1)
    pool_files = [os.path.join(work_dir, f"{split}_s{s}.parquet") for s in (2, 3)]
    stats = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for country in sorted(s1["country"].unique().to_list()):
            done = os.path.join(out_dir, f"{country}.done")
            if os.path.exists(done):
                log(f"[pairs:{split}] {country} already done")
                continue
            t0 = time.time()
            Q = s1.filter(pl.col("country") == country)
            P = pl.scan_parquet(pool_files).filter(pl.col("country") == country).collect()
            log(f"[pairs:{split}] {country}: {Q.height:,} S1 vs {P.height:,} pool records")
            if P.height == 0:
                open(done, "w").close()
                continue
            batches = []
            for qs in range(0, Q.height, q_batch):
                Qb = Q.slice(qs, q_batch)
                b = Blocker(top_k=top_k, top_k_name=top_k_name).fit(P, Qb)
                batches.append(b.query(Qb.height).with_columns(pl.col("qid") + qs))
                del b
                log(f"[pairs:{split}] {country}: blocked {min(qs + q_batch, Q.height):,}/{Q.height:,}")
            pairs = add_block_context(pl.concat(batches)).sort("qid", "rank")
            del batches
            log(f"[pairs:{split}] {country}: {pairs.height:,} candidate pairs "
                f"({pairs.height / Q.height:.2f}/entity), blocking {time.time() - t0:.0f}s")
            for i, start in enumerate(range(0, Q.height, q_chunk)):
                part = pairs.filter(pl.col("qid").is_between(start, start + q_chunk - 1))
                f = string_features(part, Q, P, executor=ex)
                part = pl.concat([part, f], how="horizontal").with_columns(
                    Q["entity_id"].gather(part["qid"]).alias("s1"),
                    P["entity_id"].gather(part["pid"]).alias("cid"),
                    pl.lit(country).alias("country"))
                part.write_parquet(os.path.join(out_dir, f"{country}_{i:03d}.parquet"))
            stats[country] = {"s1": Q.height, "pool": P.height, "pairs": pairs.height,
                              "seconds": round(time.time() - t0)}
            log(f"[pairs:{split}] {country} done in {time.time() - t0:.0f}s")
            del pairs, P, Q
            open(done, "w").close()
    return stats


def pair_files(work_dir, split):
    return sorted(glob.glob(os.path.join(work_dir, f"pairs_{split}", "*.parquet")))


def scan_pairs(work_dir, split):
    return pl.scan_parquet(pair_files(work_dir, split))


# --------------------------------------------------------------------------
# Training / validation
# --------------------------------------------------------------------------
def _xy(df, cols):
    return df.select(cols).to_numpy().astype(np.float32), df["y"].to_numpy()


def score_all(work_dir, split, stage1, oof_models=None):
    """p1 for every pair of a split. oof_models: {fold: model} used for rows of that fold."""
    out = []
    for f in pair_files(work_dir, split):
        df = pl.read_parquet(f)
        if oof_models:
            df = df.with_columns(fold_expr())
        p = predict(stage1, df.select(MODEL_FEATURES).to_numpy().astype(np.float32))
        if oof_models:
            fold = df["fold"].to_numpy()
            for k, m in oof_models.items():
                idx = np.where(fold == k)[0]
                if len(idx):
                    p[idx] = predict(m, df[idx].select(MODEL_FEATURES).to_numpy().astype(np.float32))
        out.append(df.select("s1", "cid").with_columns(pl.Series("p1", p)))
    return pl.concat(out)


def stage2_scores(work_dir, split, p1_table, stage2):
    """p2 for every pair of a split (p1_table: s1, cid, p1 for all pairs)."""
    ctx = add_stage2_features(p1_table)
    out = []
    for f in pair_files(work_dir, split):
        df = pl.read_parquet(f).join(ctx, on=["s1", "cid"], how="left")
        p2 = predict(stage2, df.select(STAGE2_FEATURES).to_numpy().astype(np.float32))
        out.append(df.select("s1", "cid", "p1").with_columns(pl.Series("p2", p2)))
    return pl.concat(out)


def train(work_dir, data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    gt = load_ground_truth(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    gt_y = gt.with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))

    def rows(folds):
        return (scan_pairs(work_dir, "train").with_columns(fold_expr())
                .filter(pl.col("fold").is_in(folds)).collect()
                .join(gt_y, on=["s1", "cid"], how="left").with_columns(pl.col("y").fill_null(0)))

    tr = rows(TRAIN_FOLDS)
    es = rows([ES_FOLD])
    log(f"[train] stage-1 rows: {tr.height:,} (pos {tr['y'].sum():,}); early-stop rows {es.height:,}")
    X_es, y_es = _xy(es, MODEL_FEATURES)

    # stage 1: one model per training fold -> out-of-fold p1 for the other fold
    oof = {}
    for k in TRAIN_FOLDS:
        part = tr.filter(pl.col("fold") != k)
        X, y = _xy(part, MODEL_FEATURES)
        oof[k] = fit_lgb(X, y, X_es, y_es)
        log(f"[train] stage-1 model (without fold {k}): {oof[k].best_iteration_} trees")
    stage1 = list(oof.values())
    del tr, es

    log("[train] scoring all train pairs with stage 1 ...")
    p1 = score_all(work_dir, "train", stage1, oof_models=oof)
    ctx = add_stage2_features(p1)

    def rows2(folds):
        r = rows(folds)
        return r.join(ctx, on=["s1", "cid"], how="left")

    tr2 = rows2(TRAIN_FOLDS)
    es2 = rows2([ES_FOLD])
    X2, y2 = _xy(tr2, STAGE2_FEATURES)
    X2e, y2e = _xy(es2, STAGE2_FEATURES)
    stage2 = fit_lgb(X2, y2, X2e, y2e)
    log(f"[train] stage-2 model: {stage2.best_iteration_} trees")
    imp = sorted(zip(STAGE2_FEATURES, stage2.feature_importances_), key=lambda x: -x[1])
    log("[train] stage-2 top features:", imp[:15])
    del tr2, es2, X2, y2

    # validation on held-out Source-1 entities, scored with the official metric
    p2 = stage2_scores(work_dir, "train", p1, stage2)
    s1_all = load(work_dir, "train", 1).select(pl.col("entity_id").alias("s1")).with_columns(fold_expr())
    val_ids = s1_all.filter(pl.col("fold").is_in(VAL_FOLDS))["s1"]
    val_truth = gt.join(pl.DataFrame({"s1": val_ids}), on="s1", how="semi")
    cand = p2.join(pl.DataFrame({"s1": val_ids}), on="s1", how="semi")
    rec = cand.join(val_truth, on=["s1", "cid"], how="semi").height / max(val_truth.height, 1)
    log(f"[val] {val_ids.len():,} held-out S1 entities; blocking recall {rec:.4f}; "
        f"{cand.height / val_ids.len():.2f} candidates/entity")
    best = (None, -1)
    results = {}
    for t in np.round(np.arange(0.20, 0.91, 0.05), 2):
        pred = decide(p2, float(t)).join(pl.DataFrame({"s1": val_ids}), on="s1", how="semi")
        m = macro_f05_frame(pred, val_truth, val_ids)
        results[float(t)] = m
        log(f"[val] threshold {t:.2f}: macro F0.5 {m['macro_f05']:.4f}  pair-P {m['precision_pairs']:.4f} "
            f"pair-R {m['recall_pairs']:.4f}  singleton acc {m['singleton_acc']:.4f}")
        if m["macro_f05"] > best[1]:
            best = (float(t), m["macro_f05"])
    # stage-1 only, for reference
    pred1 = decide(p2, 0.5, score_col="p1").join(pl.DataFrame({"s1": val_ids}), on="s1", how="semi")
    log(f"[val] stage-1 only @0.5: {macro_f05_frame(pred1, val_truth, val_ids)['macro_f05']:.4f}")
    log(f"[val] BEST threshold {best[0]:.2f} -> macro F0.5 {best[1]:.4f}")

    joblib.dump({"stage1": stage1, "stage2": stage2, "threshold": best[0]},
                os.path.join(model_dir, "entity_match_model.joblib"))
    with open(os.path.join(model_dir, "validation.json"), "w") as f:
        json.dump({"best_threshold": best[0], "best_macro_f05": best[1], "blocking_recall": rec,
                   "by_threshold": results}, f, indent=2)
    return best


# --------------------------------------------------------------------------
# Test inference
# --------------------------------------------------------------------------
def predict_test(work_dir, model_dir, out_dir, threshold=None):
    os.makedirs(out_dir, exist_ok=True)
    bundle = joblib.load(os.path.join(model_dir, "entity_match_model.joblib"))
    t = bundle["threshold"] if threshold is None else threshold
    p1 = score_all(work_dir, "test", bundle["stage1"])
    p2 = stage2_scores(work_dir, "test", p1, bundle["stage2"])
    matches = decide(p2, t)
    s1_ids = load(work_dir, "test", 1)["entity_id"]      # original file order
    write_lists(p2.select("s1", "cid"), s1_ids, os.path.join(out_dir, "candidate_pairs.tsv"),
                "candidate_entity_ids")
    write_lists(matches, s1_ids, os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids")
    n_match = matches.height
    log(f"[predict] threshold {t:.2f}: {n_match:,} matched pairs; "
        f"{p2.height / s1_ids.len():.2f} candidates/entity; "
        f"{matches['s1'].n_unique() / s1_ids.len():.3f} of entities have >=1 match")


def write_lists(pairs, s1_ids, path, col):
    agg = pairs.unique(["s1", "cid"], maintain_order=True).group_by("s1", maintain_order=True) \
               .agg(pl.col("cid").str.join(",").alias(col))
    out = pl.DataFrame({"source1_entity_id": s1_ids}).join(agg.rename({"s1": "source1_entity_id"}),
                                                           on="source1_entity_id", how="left") \
            .with_columns(pl.col(col).fill_null(""))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s, c in out.iter_rows():
            f.write(f"{s}\t{c}\n")
    log(f"wrote {path} ({out.height:,} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "prepare", "pairs", "train", "predict"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--top-k", type=int, default=12, help="candidates kept by total blocking score")
    ap.add_argument("--top-k-name", type=int, default=6, help="extra candidates kept by name score")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--threshold", type=float, default=None, help="override tuned decision threshold")
    a = ap.parse_args()
    model_dir = os.path.join(a.work_dir, "models")
    splits = ["train", "test"] if a.split == "both" else [a.split]

    if a.step in ("all", "prepare"):
        prepare(a.data_dir, a.work_dir)
    if a.step in ("all", "pairs"):
        for s in splits:
            st = build_pairs(a.work_dir, s, a.top_k, a.top_k_name, a.jobs)
            log(f"[pairs:{s}]", st)
    if a.step in ("all", "train"):
        train(a.work_dir, a.data_dir, model_dir)
    if a.step in ("all", "predict"):
        predict_test(a.work_dir, model_dir, a.out_dir, a.threshold)


if __name__ == "__main__":
    main()
