"""
Candidate generation (blocking).

Every record is decomposed into blocking keys (all scoped to one country):
  n  name token                       N  sorted pair of name tokens
  k  phonetic skeleton of name token  K  sorted pair of skeleton tokens (cross-script)
  c  first 6 chars of space-less name (domain forms: moravueares.com)
  X  name token x address word        (same business name in the same street / town)
  a  house number + following word    d  any number in the address (>= 2 digits)
  w  address word                     D  number x address word
  M  pair of address numbers          V  pair of address words

The data is built from a small vocabulary, so single words are frequent; the
combination keys (N, K, X, D, M, V) are what make a record findable.  Keys are
weighted by IDF inside the pool; keys whose bucket is larger than the per-kind
`max_df` are dropped (low information, and they would blow up the join).

For each Source-1 record every pool record sharing >= 1 key is scored with the
sum of the shared keys' weights.  Candidates = top-K by that score, plus the
top-K' by the name part of the score.  Only pool keys that also occur in some
Source-1 record are indexed (the others can never produce a candidate), which
keeps memory small.  Everything runs as vectorised Polars joins.
"""

import math

import polars as pl

KIND_WEIGHT = {"n": 1.0, "N": 1.0, "k": 0.5, "K": 0.6, "c": 1.0, "X": 0.8,
               "a": 1.5, "d": 0.8, "w": 0.5, "D": 0.8, "M": 0.8, "V": 0.6}
KIND_MAX_DF = {"n": 1000, "N": 1500, "k": 1000, "K": 1500, "c": 1500, "X": 1500,
               "a": 1500, "d": 1000, "w": 1000, "D": 1500, "M": 1500, "V": 1500}
NAME_KINDS = {"n", "N", "k", "K", "c"}
KIND_ID = {k: i for i, k in enumerate(KIND_WEIGHT)}
MAX_NAME_TOK = 5
MAX_ADDR_WORDS = 8


def _tokens(base, col, pattern, limit):
    return (base.select("rid", pl.col(col).str.extract_all(pattern).list.unique(maintain_order=True)
                        .list.head(limit).alias("t"))
                .explode("t").drop_nulls("t")
                .with_columns(pl.int_range(pl.len()).over("rid").alias("pos")))


def _finish(df, kind, expr):
    return df.select("rid", expr.hash(seed=7).alias("key")).unique()


def _pairs(tok):
    j = tok.join(tok, on="rid", suffix="_b").filter(pl.col("pos") < pl.col("pos_b"))
    return j.select("rid", pl.min_horizontal("t", "t_b").alias("x"), pl.max_horizontal("t", "t_b").alias("y"))


def iter_record_keys(df: pl.DataFrame, offset: int = 0):
    """Yields (kind, DataFrame[rid u32, key u64]) for the normalised frame df (rid = offset + row)."""
    base = df.select((pl.int_range(pl.len(), dtype=pl.UInt32) + offset).alias("rid"),
                     "nm", "sk", "cmp_a", "cmp_b", "ad")
    nt = _tokens(base, "nm", r"\S{2,}", MAX_NAME_TOK)
    st = _tokens(base, "sk", r"\S{2,}", MAX_NAME_TOK)
    aw = _tokens(base, "ad", r"\b[a-z]{3,}\b", MAX_ADDR_WORDS)
    an = _tokens(base, "ad", r"\b\d+\b", 4)

    yield "n", _finish(nt, "n", pl.lit("n") + pl.col("t"))
    yield "k", _finish(st, "k", pl.lit("k") + pl.col("t"))
    yield "N", _finish(_pairs(nt), "N", pl.col("x") + "|" + pl.col("y"))
    yield "K", _finish(_pairs(st), "K", pl.col("x") + "|" + pl.col("y"))
    xw = nt.filter(pl.col("pos") < 4).select("rid", "t").join(
        aw.filter(pl.col("pos") < 5).select("rid", pl.col("t").alias("w")), on="rid")
    yield "X", _finish(xw, "X", pl.col("t") + "|" + pl.col("w"))
    del xw
    cm = pl.concat([base.select("rid", pl.col(c).alias("t")) for c in ("cmp_a", "cmp_b")]) \
           .filter(pl.col("t").str.len_chars() >= 5)
    yield "c", _finish(cm, "c", pl.lit("c") + pl.col("t").str.slice(0, 6))
    yield "a", _finish(_tokens(base, "ad", r"\b\d+ [a-z]{2,}", 4), "a", pl.lit("a") + pl.col("t"))
    yield "d", _finish(_tokens(base, "ad", r"\b\d{2,}\b", 6), "d", pl.lit("d") + pl.col("t"))
    yield "w", _finish(aw, "w", pl.lit("w") + pl.col("t"))
    # address combinations: a specific full address stays rare even when its parts are common
    dw = an.filter(pl.col("pos") < 3).select("rid", "t").join(
        aw.filter(pl.col("pos") < 5).select("rid", pl.col("t").alias("w")), on="rid")
    yield "D", _finish(dw, "D", pl.col("t") + "|" + pl.col("w"))
    del dw
    yield "M", _finish(_pairs(an), "M", pl.col("x") + "|" + pl.col("y"))
    yield "V", _finish(_pairs(aw.filter(pl.col("pos") < 5)), "V", pl.col("x") + "|" + pl.col("y"))


def _chunked_keys(df: pl.DataFrame, chunk: int):
    """{kind: DataFrame[rid, key]} built in record chunks to bound peak memory."""
    parts = {k: [] for k in KIND_ID}
    for start in range(0, df.height, chunk):
        for kind, k in iter_record_keys(df.slice(start, chunk), offset=start):
            parts[kind].append(k)
    return parts


class Blocker:
    def __init__(self, top_k: int = 12, top_k_name: int = 6, chunk: int = 6000, kind_weight=None,
                 kind_max_df=None, record_chunk: int = 500000):
        self.top_k = top_k
        self.top_k_name = top_k_name
        self.chunk = chunk
        self.kind_weight = dict(kind_weight or KIND_WEIGHT)
        self.kind_max_df = dict(kind_max_df or KIND_MAX_DF)
        self.record_chunk = record_chunk

    def fit(self, pool: pl.DataFrame, queries: pl.DataFrame):
        """Index the Source-2/3 pool of ONE country for the given Source-1 queries."""
        self.n = pool.height
        qparts = _chunked_keys(queries, self.record_chunk)
        self.qkeys = pl.concat([pl.concat(v).with_columns(pl.lit(KIND_ID[k], dtype=pl.UInt8).alias("kind"))
                                for k, v in qparts.items() if v]).rename({"rid": "qid"})
        del qparts
        wanted = self.qkeys.select("key").unique()
        pparts = {k: [] for k in KIND_ID}
        for start in range(0, pool.height, self.record_chunk):
            for kind, k in iter_record_keys(pool.slice(start, self.record_chunk), offset=start):
                pparts[kind].append(k.join(wanted, on="key", how="semi"))
        stats, postings = [], []
        for kind, v in pparts.items():
            if not v:
                continue
            pk = pl.concat(v)
            dfreq = pk.group_by("key").agg(pl.len().cast(pl.UInt32).alias("df")) \
                      .filter(pl.col("df") <= max(self.kind_max_df.values())) \
                      .with_columns(pl.lit(KIND_ID[kind], dtype=pl.UInt8).alias("kind"))
            stats.append(dfreq)
            postings.append(pk.join(dfreq.select("key"), on="key", how="semi").rename({"rid": "pid"}))
        del pparts
        self.stats = pl.concat(stats)
        self.pool_keys = pl.concat(postings)
        return self

    def _weights(self):
        logn = math.log(self.n + 1)
        kinds = list(KIND_ID)
        cap = pl.col("kind").replace_strict({KIND_ID[k]: self.kind_max_df[k] for k in kinds}, return_dtype=pl.UInt32)
        kw = pl.col("kind").replace_strict({KIND_ID[k]: self.kind_weight[k] for k in kinds}, return_dtype=pl.Float32)
        is_name = pl.col("kind").is_in([KIND_ID[k] for k in NAME_KINDS])
        return self.stats.filter(pl.col("df") <= cap).select(
            "key", (kw * (logn - (pl.col("df").cast(pl.Float32) + 1).log())).cast(pl.Float32).alias("w"),
            is_name.alias("is_name"))

    def query(self, n_queries: int) -> pl.DataFrame:
        """Candidates for the queries given to fit(): (qid, pid, score, name_score, addr_score, n_keys, rank)."""
        weights = self._weights()
        qk = self.qkeys.select("qid", "key").join(weights, on="key", how="inner")
        out = []
        for start in range(0, n_queries, self.chunk):
            sub = qk.filter(pl.col("qid").is_between(start, start + self.chunk - 1))
            j = sub.join(self.pool_keys, on="key", how="inner")
            s = j.group_by("qid", "pid").agg(
                pl.col("w").sum().alias("score"),
                pl.col("w").filter(pl.col("is_name")).sum().alias("name_score"),
                pl.col("w").filter(~pl.col("is_name")).sum().alias("addr_score"),
                pl.len().cast(pl.UInt16).alias("n_keys"),
            )
            del j
            # candidates = top-K by total score  UNION  top-K' by name score
            s = s.with_columns(
                pl.col("score").rank("ordinal", descending=True).over("qid").cast(pl.UInt32).alias("rank"),
                pl.struct(pl.col("name_score"), pl.col("score")).rank("ordinal", descending=True)
                  .over("qid").cast(pl.UInt32).alias("_nr"),
            ).filter((pl.col("rank") <= self.top_k) | (pl.col("_nr") <= self.top_k_name)).drop("_nr")
            out.append(s)
        if not out:
            return pl.DataFrame(schema={"qid": pl.UInt32, "pid": pl.UInt32, "score": pl.Float32,
                                        "name_score": pl.Float32, "addr_score": pl.Float32,
                                        "n_keys": pl.UInt16, "rank": pl.UInt32})
        return pl.concat(out)
