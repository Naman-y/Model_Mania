"""Normalise every source file once and cache it as parquet (work_dir/{split}_s{1,2,3}.parquet)."""

import argparse
import os
import time

import polars as pl

from normalize import normalize_frame, read_tsv


def prepare(data_dir: str, work_dir: str, splits=("train", "test"), force: bool = False):
    os.makedirs(work_dir, exist_ok=True)
    for split in splits:
        for s in (1, 2, 3):
            out = os.path.join(work_dir, f"{split}_s{s}.parquet")
            if os.path.exists(out) and not force:
                continue
            t = time.time()
            df = read_tsv(os.path.join(data_dir, split, f"{split}_source{s}.tsv"))
            normalize_frame(df).write_parquet(out)
            print(f"[prepare] {out}: {df.height:,} rows in {time.time() - t:.0f}s", flush=True)
            del df


def load(work_dir: str, split: str, s: int) -> pl.DataFrame:
    return pl.read_parquet(os.path.join(work_dir, f"{split}_s{s}.parquet"))


def load_pool(work_dir: str, split: str) -> pl.DataFrame:
    return pl.concat([load(work_dir, split, 2), load(work_dir, split, 3)])


def load_ground_truth(path: str) -> pl.DataFrame:
    """(s1, cid) pairs of true matches."""
    gt = read_tsv(path)
    return gt.select(pl.col("source1_entity_id").alias("s1"),
                     pl.col("matched_entity_ids").fill_null("").str.split(",").alias("cid")) \
             .explode("cid").filter(pl.col("cid").str.len_chars() > 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    prepare(a.data_dir, a.work_dir, force=a.force)
