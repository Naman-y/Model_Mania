"""
Pairwise similarity features for (Source-1 record, candidate) pairs.

String similarities use RapidFuzz (C++), evaluated in parallel worker
processes over chunks of pairs.  All features are country-agnostic so the
model transfers to countries unseen in training (France).
"""

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

REC_COLS = ["nm", "nm_a", "nm_b", "cmp_a", "cmp_b", "sk", "ad", "nums"]

STRING_FEATURES = [
    "nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "alt_tset", "alt_ratio", "cmp_ratio", "cmp_jw",
    "cmp_contain", "initials", "tok_jacc", "tok_q_cov", "tok_c_cov", "sk_jacc", "sk_ratio", "sk_tset",
    "ntok_q", "ntok_c",
    "ad_tset", "ad_tsort", "ad_partial", "ad_word_jacc", "ad_word_c_cov", "num_q", "num_c", "num_common",
    "num_jacc", "num_c_subset", "num_first_eq", "num_long_common", "num_conflict",
]
FLAG_FEATURES = ["q_has_addr", "c_has_addr", "c_is_domain", "c_is_native", "q_is_native", "is_s2"]
BLOCK_FEATURES = ["score", "name_score", "addr_score", "n_keys", "rank", "name_rank", "score_rel",
                  "score_gap", "n_cand", "pid_nq", "pid_rank", "pid_score_rel"]
MODEL_FEATURES = BLOCK_FEATURES + STRING_FEATURES + FLAG_FEATURES


def _jacc(a, b):
    if not a or not b:
        return -1.0
    return len(a & b) / len(a | b)


def _pair_features(q, c):
    q_nm, q_a, q_b, q_ca, q_cb, q_sk, q_ad, q_nums = q
    c_nm, c_a, c_b, c_ca, c_cb, c_sk, c_ad, c_nums = c
    f = []
    if q_nm and c_nm:
        f += [fuzz.ratio(q_nm, c_nm), fuzz.token_sort_ratio(q_nm, c_nm),
              fuzz.token_set_ratio(q_nm, c_nm), fuzz.partial_ratio(q_nm, c_nm)]
    else:
        f += [0.0, 0.0, 0.0, 0.0]

    q_alts = [x for x in (q_a, q_b) if x]
    c_alts = [x for x in (c_a, c_b) if x]
    q_cmps = [x for x in (q_ca, q_cb) if x]
    c_cmps = [x for x in (c_ca, c_cb) if x]
    f.append(max((fuzz.token_set_ratio(x, y) for x in q_alts for y in c_alts), default=0.0))
    f.append(max((fuzz.ratio(x, y) for x in q_alts for y in c_alts), default=0.0))
    f.append(max((fuzz.ratio(x, y) for x in q_cmps for y in c_cmps), default=0.0))
    f.append(max((JaroWinkler.similarity(x, y) for x in q_cmps for y in c_cmps), default=0.0))
    contain = 0.0
    for x in q_cmps:
        for y in c_cmps:
            if len(x) >= 4 and len(y) >= 4 and (x in y or y in x):
                contain = 1.0
    f.append(contain)
    # "mc.com" style initials of one name matching the other (short) name
    ini = 0.0
    q_ini = "".join(t[0] for t in q_a.split()) if q_a else ""
    c_ini = "".join(t[0] for t in c_a.split()) if c_a else ""
    if len(q_ini) >= 2 and c_ca and len(c_ca) <= 6 and q_ini.startswith(c_ca):
        ini = 1.0
    if len(c_ini) >= 2 and q_ca and len(q_ca) <= 6 and c_ini.startswith(q_ca):
        ini = 1.0
    f.append(ini)
    qt, ct = set(q_nm.split()), set(c_nm.split())
    inter = len(qt & ct)
    f.append(_jacc(qt, ct))
    f.append(inter / len(qt) if qt else -1.0)
    f.append(inter / len(ct) if ct else -1.0)
    qs, cs = set(q_sk.split()), set(c_sk.split())
    f.append(_jacc(qs, cs))
    f.append(fuzz.ratio(" ".join(sorted(qs)), " ".join(sorted(cs))))
    f.append(fuzz.token_set_ratio(q_sk, c_sk) if q_sk and c_sk else 0.0)
    f.append(len(qt))
    f.append(len(ct))

    if q_ad and c_ad:
        f += [fuzz.token_set_ratio(q_ad, c_ad), fuzz.token_sort_ratio(q_ad, c_ad), fuzz.partial_ratio(q_ad, c_ad)]
        qw = {t for t in q_ad.split() if not t.isdigit()}
        cw = {t for t in c_ad.split() if not t.isdigit()}
        f.append(_jacc(qw, cw))
        f.append(len(qw & cw) / len(cw) if cw else -1.0)
    else:
        f += [-1.0, -1.0, -1.0, -1.0, -1.0]

    qn = q_nums.split() if q_nums else []
    cn = c_nums.split() if c_nums else []
    qns, cns = set(qn), set(cn)
    common = qns & cns
    f += [len(qns), len(cns), len(common), _jacc(qns, cns)]
    f.append(1.0 if cns and cns <= qns else (0.0 if cns else -1.0))
    f.append(1.0 if (qn and cn and qn[0] == cn[0]) else (0.0 if (qn and cn) else -1.0))
    f.append(max((len(x) for x in common), default=0))
    # both have a "long" number (house no / PIN) and none of them agree
    ql = {x for x in qns if len(x) >= 3}
    cl = {x for x in cns if len(x) >= 3}
    f.append(1.0 if (ql and cl and not (ql & cl)) else 0.0)
    return f


def _chunk_features(args):
    q_rows, c_rows = args
    return np.array([_pair_features(q, c) for q, c in zip(q_rows, c_rows)], dtype=np.float32)


def add_block_context(pairs: pl.DataFrame) -> pl.DataFrame:
    """Context features over ALL pairs of a country (needs every S1 that competes for a pid)."""
    return pairs.with_columns(
        pl.col("name_score").rank("ordinal", descending=True).over("qid").cast(pl.UInt32).alias("name_rank"),
        (pl.col("score") / pl.col("score").max().over("qid")).alias("score_rel"),
        (pl.col("score") - pl.col("score").max().over("qid")).alias("score_gap"),
        pl.len().over("qid").cast(pl.UInt16).alias("n_cand"),
        pl.len().over("pid").cast(pl.UInt16).alias("pid_nq"),
        pl.col("score").rank("ordinal", descending=True).over("pid").cast(pl.UInt16).alias("pid_rank"),
        (pl.col("score") / pl.col("score").max().over("pid")).alias("pid_score_rel"),
    )


def string_features(pairs: pl.DataFrame, q: pl.DataFrame, pool: pl.DataFrame, executor=None,
                    chunk: int = 20000) -> pl.DataFrame:
    """
    pairs: (qid, pid, ...) ; q / pool: normalised frames of one country indexed by qid / pid.
    Returns string + flag features aligned with `pairs` rows.
    """
    qi = pairs["qid"].to_numpy()
    pi = pairs["pid"].to_numpy()
    qsel = q.select(REC_COLS)[qi]
    psel = pool.select(REC_COLS)[pi]
    q_rows = qsel.rows()
    c_rows = psel.rows()
    del qsel, psel
    jobs = [(q_rows[i:i + chunk], c_rows[i:i + chunk]) for i in range(0, len(q_rows), chunk)]
    if executor is not None and len(jobs) > 1:
        mats = list(executor.map(_chunk_features, jobs))
    else:
        mats = [_chunk_features(j) for j in jobs]
    X = np.vstack(mats) if mats else np.zeros((0, len(STRING_FEATURES)), dtype=np.float32)
    feats = pl.DataFrame(X, schema=STRING_FEATURES)
    flags = pl.DataFrame({
        "q_has_addr": q["has_addr"].to_numpy()[qi],
        "c_has_addr": pool["has_addr"].to_numpy()[pi],
        "c_is_domain": pool["is_domain"].to_numpy()[pi],
        "c_is_native": pool["is_native"].to_numpy()[pi],
        "q_is_native": q["is_native"].to_numpy()[qi],
        "is_s2": pool["entity_id"].str.starts_with("S2-").to_numpy()[pi],
    }).cast(pl.Float32)
    return pl.concat([feats, flags], how="horizontal")
