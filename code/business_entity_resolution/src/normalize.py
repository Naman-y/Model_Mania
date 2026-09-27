"""
Vectorised record normalisation (Polars expressions only, so it scales to the
~10M-row candidate pools without Python loops).

Per record we produce:
  nm      normalised name tokens of the whole name (legal suffixes / noise removed)
  nm_a    primary name alternative   (text before "aka", "dba", "|", "t/a", ...)
  nm_b    secondary name alternative (text after the separator, "" if none)
  cmp_a   nm_a with spaces removed   (matches domain forms: moravueares.com)
  cmp_b   nm_b with spaces removed
  sk      phonetic skeleton of nm tokens (script independent: Hindi / Tamil /
          Malayalam names transliterate to the same skeleton as Latin names)
  ad      normalised address tokens (abbreviations canonicalised)
  nums    space separated numbers found in the address (leading zeros removed)
  has_addr / is_domain / is_native flags
"""

import polars as pl

# --------------------------------------------------------------------------
# Indic scripts -> Latin.  All major Indic Unicode blocks share the same
# layout, so one offset table (relative to the block start) covers
# Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and
# Malayalam.  Consonants get a marker so the inherent vowel can be resolved
# with a couple of regex passes afterwards.
# --------------------------------------------------------------------------
_INDIC_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_C = "\x01"   # consonant marker (inherent vowel pending)
_V = "\x02"   # dependent vowel sign follows
_H = "\x03"   # virama (kills inherent vowel)

_OFFSETS = {
    0x01: "n", 0x02: "n", 0x03: "h",
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x3C: "", 0x3D: "", 0x4D: _H, 0x4E: "", 0x4F: "", 0x50: "om", 0x51: "", 0x52: "", 0x53: "", 0x54: "",
    0x55: "", 0x56: "", 0x57: "", 0x60: "ri", 0x61: "li", 0x62: "", 0x63: "", 0x64: " ", 0x65: " ",
}
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_MATRAS = {
    0x3E: "aa", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
}


def _build_indic_map():
    pats, reps = [], []
    for base in _INDIC_BLOCKS:
        for off, rep in _OFFSETS.items():
            pats.append(chr(base + off)); reps.append(rep)
        for off, rep in _CONS.items():
            pats.append(chr(base + off)); reps.append(rep + _C)
        for off, rep in _MATRAS.items():
            pats.append(chr(base + off)); reps.append(_V + rep)
        for d in range(10):
            pats.append(chr(base + 0x66 + d)); reps.append(str(d))
    extras = {
        "ൺ": "n", "ൻ": "n", "ർ": "r", "ൽ": "l", "ൾ": "l", "ൿ": "k",  # Malayalam chillu
        "ৎ": "t", "ৰ": "r", "ৱ": "w",  # Bengali / Assamese
        "ੰ": "n", "ੱ": "", "ௗ": "", "ஃ": "",  # Gurmukhi tippi/addak, Tamil marks
        "‌": "", "‍": "",  # ZWNJ / ZWJ
    }
    for k, v in extras.items():
        pats.append(k); reps.append(v)
    return pats, reps


_INDIC_PATS, _INDIC_REPS = _build_indic_map()


def translit(e: pl.Expr) -> pl.Expr:
    """Indic -> Latin, accent folding, lowercase."""
    e = e.fill_null("").str.replace_many(_INDIC_PATS, _INDIC_REPS)
    e = (e.str.replace_all(_C + _V, "", literal=True)      # consonant + vowel sign
          .str.replace_all(_C + _H, "", literal=True)      # consonant + virama
          .str.replace_all(_C + r"(\s|$|[^\p{L}])", "$1")  # word-final schwa deletion
          .str.replace_all(_C, "a", literal=True)
          .str.replace_all(r"[\x02\x03]", ""))
    e = e.str.normalize("NFKD").str.replace_all(r"\p{M}", "")
    return e.str.to_lowercase()


# --------------------------------------------------------------------------
# Name normalisation
# --------------------------------------------------------------------------
_ALT_SEP = (r"\s+(?:aka|a\.k\.a\.?|dba|d/b/a|doing business as|t/a|trading as|"
            r"formerly known as|formerly|fka|f/k/a)\b\s*:?\s*|\s*\|\s*")
_TLD = r"\.(?:com|co\.in|in|org|net|co|fr|biz|info|us|io)\b"

LEGAL_WORDS = [
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "ltd", "limited",
    "pvt", "private", "prv", "llp", "lp", "pllc", "pc", "plc", "sarl", "sas", "sasu", "sa",
    "eurl", "sci", "snc", "selarl", "gmbh", "ag", "the", "and", "of", "et", "le", "la", "les",
    "des", "du", "de", "www", "http", "https", "public",
    # transliterated native-script forms of private / limited / LLP / "pvt. ltd."
    "praaivet", "praivet", "praaibhet", "piraivet", "praivarr", "praivat", "limitet", "limatid",
    "limirrad", "elaelapi", "elelpi", "ailaailapi", "praa", "li",
    # honorific prefixes added as noise
    "mr", "mrs", "smt", "messrs",
]
_LEGAL_PATS = [f" {w} " for w in LEGAL_WORDS]
# long legal words whose typos ("prviate", "lmited") should also be removed
_LEGAL_FUZZY = ["private", "limited", "corporation", "incorporated", "company"]
# leetspeak noise inside words: cardi0logy, 5hah, 6lobal
_LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"}


def _clean_name_part(e: pl.Expr) -> pl.Expr:
    e = (e.str.replace_all(r"(?:https?://)?www\.", " ")
          .str.replace_all(_TLD, " ")
          .str.replace_all(r"\bm/s\b|\bid\s*\d+|\b\d{5,}\b", " ")
          .str.replace_all(r"&|\+", " and ")
          .str.replace_all(r"'s\b", "s")
          .str.replace_all(r"[^\p{L}\p{N}]+", " "))
    for d, ch in _LEET.items():
        e = e.str.replace_all(rf"(\p{{L}}){d}", f"${{1}}{ch}").str.replace_all(rf"\b{d}(\p{{L}})", f"{ch}${{1}}")
    e = pl.lit(" ") + e.str.replace_all(" ", "  ") + pl.lit(" ")
    e = e.str.replace_many(_LEGAL_PATS, [" "] * len(_LEGAL_PATS))
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def _legal_typos(names: pl.Series) -> list:
    """Tokens within edit distance 1-2 of a long legal word (e.g. 'prviate', 'lmited')."""
    from rapidfuzz.distance import OSA
    toks = names.str.split(" ").explode().unique().drop_nulls()
    toks = toks.filter(toks.str.len_chars() >= 5).to_list()
    out = []
    for t in toks:
        for w in _LEGAL_FUZZY:
            k = 1 if len(w) < 8 else 2
            if t != w and abs(len(t) - len(w)) <= k and OSA.distance(t, w, score_cutoff=k) <= k:
                out.append(t)
                break
    return out


def _drop_tokens(e: pl.Expr, toks: list) -> pl.Expr:
    if not toks:
        return e
    pats = [f" {t} " for t in toks]
    e = pl.lit(" ") + e.str.replace_all(" ", "  ") + pl.lit(" ")
    e = e.str.replace_many(pats, [" "] * len(pats))
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


# Phonetic skeleton: soundex-like classes over the full token, so that
# "southern properties" and "सदर्न प्रॉपर्टीज" -> "sadarn praapartij" both map to
# "krtrn prprtk".
_SK_DIGRAPH = (["ph", "gh", "kh", "th", "dh", "bh", "sh", "ch", "jh", "ck", "wh"],
               ["f", "g", "k", "t", "d", "b", "s", "c", "j", "k", "w"])
_SK_CLASS = (list("bfpvwcgjkqsxzdtmn"), list("ppppp" "kkkkkkkk" "tt" "nn"))


def skeleton(e: pl.Expr) -> pl.Expr:
    e = e.str.replace_all(r"ti([aou])", "s$1")      # -tion / -tial ~ "shan"
    e = e.str.replace_many(*_SK_DIGRAPH)
    e = e.str.replace_all(r"\b[aeiou]", "A")       # keep vowel-initial marker
    e = e.str.replace_all(r"[aeiouyh]", "")
    e = e.str.replace_many(*_SK_CLASS)
    for _ in range(3):
        e = e.str.replace_many(["pp", "kk", "tt", "nn", "ll", "rr"], ["p", "k", "t", "n", "l", "r"])
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


# --------------------------------------------------------------------------
# Address normalisation
# --------------------------------------------------------------------------
_ADDR_CANON = {
    "st": ["street", "str", "saint", "st", "sainte", "ste"],
    "av": ["avenue", "ave", "av", "avn"],
    "rd": ["road", "rd"],
    "dr": ["drive", "dr", "drv"],
    "ln": ["lane", "ln"],
    "ct": ["court", "ct", "crt"],
    "cir": ["circle", "cir", "crcl"],
    "bd": ["boulevard", "blvd", "bd", "bvd", "boul"],
    "pkwy": ["parkway", "pkwy", "pky"],
    "hwy": ["highway", "hwy"],
    "pl": ["place", "pl"],
    "sq": ["square", "sq"],
    "ter": ["terrace", "ter", "terr"],
    "trl": ["trail", "trl"],
    "way": ["way", "wy"],
    "rue": ["rue", "r"],
    "chemin": ["chemin", "chem", "ch"],
    "imp": ["impasse", "imp"],
    "allee": ["allee", "all"],
    "rte": ["route", "rte"],
    "fbg": ["faubourg", "fbg", "fg"],
    "qu": ["quai", "qu"],
    "mt": ["mount", "mt", "mont"],
    "ft": ["fort", "ft"],
    "pt": ["point", "pt"],
    "n": ["north", "n", "nord"],
    "s": ["south", "s", "sud"],
    "e": ["east", "e", "est"],
    "w": ["west", "w", "ouest"],
    "nagar": ["nagar", "ngr"],
    "colony": ["colony", "col"],
    "sector": ["sector", "sec"],
    "bldg": ["building", "bldg", "bldng"],
    # Indian states: English, code and transliterated native-script forms
    "mh": ["maharashtra", "mahaaraashtr"], "dl": ["delhi", "dilli"],
    "up": ["uttar pradesh"], "ka": ["karnataka", "karnaatak"],
    "tn": ["tamil nadu", "tamilnadu", "tamizhnaatu"], "gj": ["gujarat", "gujaraat"],
    "wb": ["west bengal", "w bengal", "pashchimabang"],
    "kl": ["kerala", "keralam", "keralan"], "ts": ["telangana", "telangaan", "tg"],
    "ap": ["andhra pradesh", "aandhrapradesh"], "rj": ["rajasthan", "raajasthaan"],
    "mp": ["madhya pradesh", "madhy pradesh"], "hr": ["haryana", "hariyaanaa"],
    "pb": ["punjab", "panjaab"], "br": ["bihar", "bihaar"],
    "or": ["odisha", "orissa", "odishaa", "od"],
    "bangalore": ["bengaluru"], "mumbai": ["bombay"], "kolkata": ["calcutta"],
    "gurgaon": ["gurugram"], "ahmedabad": ["ahmadabad"],
    "1": ["first"], "2": ["second"], "3": ["third"], "4": ["fourth"], "5": ["fifth"],
    "6": ["sixth"], "7": ["seventh"], "8": ["eighth"], "9": ["ninth"], "10": ["tenth"],
}
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia",
}
# Tokens carrying no identity information (unit / landmark / filler words).
_ADDR_DROP = ["unit", "apt", "apartment", "suite", "ste", "fl", "floor", "flr", "no", "num", "number",
              "null", "nan", "none", "door", "h", "hno", "plot", "shop", "flat", "kh", "khasra",
              "near", "nr", "opp", "opposite", "behind", "beside", "next", "to", "the", "of", "and",
              "po", "box", "pmb", "bis", "c", "o", "d", "du", "de", "la", "le", "des", "et"]


def _word_map(pairs):
    """-> (multi-word patterns, single-word patterns), each (pats, reps), longest first."""
    multi, single = ([], []), ([], [])
    for canon, variants in pairs:
        for v in variants:
            tgt = multi if " " in v else single
            tgt[0].append(f" {v} "); tgt[1].append(f" {canon} ")
    for m in (multi, single):
        order = sorted(range(len(m[0])), key=lambda i: -len(m[0][i]))
        m[0][:] = [m[0][i] for i in order]; m[1][:] = [m[1][i] for i in order]
    return multi, single


_STATE_MAP = _word_map((k, [v]) for k, v in US_STATES.items())
_CANON_MAP = _word_map(_ADDR_CANON.items())
_DROP_PATS = [f" {w} " for w in _ADDR_DROP]


def _pad(e):
    return pl.lit(" ") + e.str.replace_all(" ", "  ") + pl.lit(" ")


def _replace_words(e, word_map):
    """Whole-word replacement. Multi-word phrases first (single-space padding), then
    single words (double-space padding so adjacent words can all match)."""
    (mp, mr), (sp, sr) = word_map
    if mp:
        e = pl.lit(" ") + e + pl.lit(" ")
        for _ in range(2):
            e = e.str.replace_many(mp, mr)
        e = e.str.replace_all(r"\s+", " ").str.strip_chars()
    e = _pad(e).str.replace_many(sp, sr)
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def _norm_address(e: pl.Expr, country: pl.Expr) -> pl.Expr:
    e = translit(e)
    e = (e.str.replace_all(r"\bn\s*[°º]", " ")
          .str.replace_all(r"\b(?:p\.?\s?o\.?\s*box|pmb|post box)\s*#?\s*\d+", " ")
          .str.replace_all(r"(\d+)\s*(?:st|nd|rd|th)\b", "$1")
          .str.replace_all(r"[^\p{L}\p{N}]+", " ")
          .str.replace_all(r"(\d)(\p{L})", "$1 $2")
          .str.replace_all(r"(\p{L})(\d)", "$1 $2")
          .str.replace_all(r"\b0+(\d)", "$1"))
    e = e.str.replace_all(r"\s+", " ").str.strip_chars()
    # US state names -> 2 letter code (only for US records; "in"/"or" are real words elsewhere)
    e = pl.when(country == "us").then(_replace_words(e, _STATE_MAP)).otherwise(e)
    e = _replace_words(e, _CANON_MAP)
    e = _pad(e)
    for _ in range(2):
        e = e.str.replace_many(_DROP_PATS, [" "] * len(_DROP_PATS))
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


# --------------------------------------------------------------------------
def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Input columns: entity_id, business_name, business_address, country."""
    df = df.rename({c: c.lstrip("+").strip() for c in df.columns})
    country = pl.col("country").fill_null("unknown").str.strip_chars().str.to_lowercase()
    raw_name = pl.col("business_name").fill_null("")
    name_t = translit(raw_name)
    out = df.select(
        pl.col("entity_id"),
        country.alias("country"),
        name_t.alias("_nt"),
        raw_name.str.contains(r"[ऀ-ൿ]").alias("is_native"),
        raw_name.str.contains(r"(?i)\.(?:com|co\.in|in|org|net|co|fr|biz|info)\b").alias("is_domain"),
        pl.col("business_address").is_not_null().alias("has_addr"),
        _norm_address(pl.col("business_address"), country).alias("ad"),
    )
    parts = pl.col("_nt").str.replace(_ALT_SEP, "\x00").str.split_exact("\x00", 1)
    out = out.with_columns(parts.struct.field("field_0").alias("_a"),
                           parts.struct.field("field_1").fill_null("").alias("_b"))
    out = out.with_columns(
        _clean_name_part(pl.col("_nt").str.replace_all(_ALT_SEP, " ")).alias("nm"),
        _clean_name_part(pl.col("_a")).alias("nm_a"),
        _clean_name_part(pl.col("_b")).alias("nm_b"),
    ).drop("_nt", "_a", "_b")
    typos = _legal_typos(out["nm"])
    out = out.with_columns(_drop_tokens(pl.col(c), typos).alias(c) for c in ("nm", "nm_a", "nm_b"))
    out = out.with_columns(
        pl.col("nm_a").str.replace_all(" ", "").alias("cmp_a"),
        pl.col("nm_b").str.replace_all(" ", "").alias("cmp_b"),
        skeleton(pl.col("nm")).alias("sk"),
        pl.col("ad").str.extract_all(r"\b\d+\b").list.join(" ").alias("nums"),
    )
    return out


def read_tsv(path: str, n_rows=None) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, n_rows=n_rows)
    return df.rename({c: c.lstrip("+").strip() for c in df.columns})
