#!/usr/bin/env python
"""
Inference on the full training Source1 entities using the trained model.
Generates candidate_pairs_train.tsv and matching_results_train.tsv preserving original order.
"""
import os, sys, gc
import numpy as np
import polars as pl
import joblib
from tqdm import tqdm
import re, unicodedata, subprocess
from collections import defaultdict
import jellyfish

# --- Utility functions (same as solution.py) ---
LEGAL_SUFFIXES = {
    'corporation', 'corp', 'incorporated', 'inc', 'limited', 'ltd',
    'private limited', 'pvt ltd', 'pvt', 'private', 'llc', 'llp',
    'co', 'company', 'gmbh', 'sa', 'sarl', 'plc', 'enterprise', 'enterprises'
}
RE_NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9\s]')
RE_STRIP_ALL = re.compile(r'[^a-zA-Z0-9]')
RE_MULTIPLE_SPACES = re.compile(r'\s+')
RE_AMPERSAND = re.compile(r'\s*\u0026\s*')
RE_PIN_6DIGIT = re.compile(r'\b(\d{3})[\s-]?(\d{3})\b')
RE_PIN_5DIGIT = re.compile(r'\b(\d{5})\b')
ADDRESS_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'hwy': 'highway', 'apt': 'apartment',
    'ste': 'suite', 'bldg': 'building', 'fl': 'floor', 'opp': 'opposite', 'nr': 'near'
}
STOPWORDS = {'the', 'and', 'of', 'in', 'for', 'on', 'at', 'to', 'a', 'an', 'is', 'group', 'india', 'international', 'global'}

def canonicalize_country(country_str: str) -> str:
    if not country_str or not isinstance(country_str, str):
        return "UNKNOWN"
    c = country_str.strip().lower()
    c = RE_STRIP_ALL.sub('', c)
    if c in {'us', 'usa', 'unitedstates', 'unitedstatesofamerica'}:
        return "US"
    if c in {'india', 'ind', 'in'}:
        return "India"
    if c in {'france', 'fr'}:
        return "France"
    return country_str.strip().title()

def transliterate_to_latin(text: str) -> str:
    if not text:
        return ""
    if all(ord(ch) < 128 for ch in text):
        return text
    try:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    except Exception:
        return text

def clean_business_name(name: str):
    if not name or not isinstance(name, str):
        return ("", "", "")
    text = transliterate_to_latin(name).lower()
    text = RE_AMPERSAND.sub(' and ', text)
    text = RE_NON_ALPHANUM.sub(' ', text)
    tokens = [t for t in RE_MULTIPLE_SPACES.sub(' ', text).strip().split() if t]
    cleaned_name = " ".join(tokens)
    joined = " " + cleaned_name + " "
    for suffix in ['private limited', 'pvt ltd']:
        if joined.endswith(f" {suffix} "):
            joined = joined[:-len(suffix)-2]
    tokens = joined.strip().split()
    filtered_tokens = [tok for tok in tokens if tok not in LEGAL_SUFFIXES]
    stripped_name = " ".join(filtered_tokens) if filtered_tokens else cleaned_name
    metaphone_tokens = [jellyfish.metaphone(tok) for tok in (filtered_tokens[:3] if filtered_tokens else tokens[:3])]
    metaphone_key = " ".join([m for m in metaphone_tokens if m])
    return cleaned_name, stripped_name, metaphone_key

def extract_pin(address: str, country: str = "UNKNOWN"):
    if not address or not isinstance(address, str):
        return None
    c = country.upper()
    if c == "INDIA":
        m = RE_PIN_6DIGIT.search(address)
        if m:
            return m.group(1) + m.group(2)
    elif c in {"US", "FRANCE"}:
        m = RE_PIN_5DIGIT.search(address)
        if m:
            return m.group(1)
    m6 = RE_PIN_6DIGIT.search(address)
    if m6:
        return m6.group(1) + m6.group(2)
    m5 = RE_PIN_5DIGIT.search(address)
    if m5:
        return m5.group(1)
    return None

def extract_city_state(address: str):
    if not address or not isinstance(address, str):
        return (None, None)
    parts = [p.strip().lower() for p in address.split(',') if p.strip()]
    if len(parts) < 2:
        return (None, None)
    candidate_city = RE_NON_ALPHANUM.sub(' ', parts[-2]).strip()
    candidate_state = RE_NON_ALPHANUM.sub(' ', parts[-1]).strip()
    if any(ch.isdigit() for ch in candidate_city) or any(ch.isdigit() for ch in candidate_state):
        return (None, None)
    if not candidate_city or not candidate_state:
        return (None, None)
    if candidate_city == candidate_state or set(candidate_city.split()) == set(candidate_state.split()):
        return (None, None)
    return (candidate_city, candidate_state)

def clean_address(address: str) -> str:
    if not address or not isinstance(address, str):
        return ""
    text = transliterate_to_latin(address).lower()
    text = RE_AMPERSAND.sub(' and ', text)
    text = RE_NON_ALPHANUM.sub(' ', text)
    tokens = text.split()
    return " ".join([ADDRESS_ABBR.get(tok, tok) for tok in tokens])

class CandidateBlocker:
    def __init__(self, max_candidates_per_entity: int = 30):
        self.max_candidates = max_candidates_per_entity
        self.token_index = defaultdict(lambda: defaultdict(list))
        self.pin_index = defaultdict(lambda: defaultdict(list))
        self.phonetic_index = defaultdict(lambda: defaultdict(list))
    def index_candidates(self, candidate_ids, countries, stripped_names, pins, metaphone_keys):
        for c_id, country, s_name, pin, meta in zip(candidate_ids, countries, stripped_names, pins, metaphone_keys):
            if pin:
                if len(self.pin_index[country][pin]) < 1000:
                    self.pin_index[country][pin].append(c_id)
                if len(pin) >= 3 and len(self.pin_index[country][pin[:3]]) < 1000:
                    self.pin_index[country][pin[:3]].append(c_id)
            tokens = [t for t in s_name.split() if len(t) >= 3 and t not in STOPWORDS]
            for tok in tokens:
                if len(self.token_index[country][tok]) < 2000:
                    self.token_index[country][tok].append(c_id)
            if meta:
                first_meta = meta.split()[0]
                if len(first_meta) >= 3 and len(self.phonetic_index[country][first_meta]) < 1000:
                    self.phonetic_index[country][first_meta].append(c_id)
    def find_candidates_for_entity(self, country: str, stripped_name: str, pin, metaphone_key: str):
        candidate_scores = defaultdict(int)
        if pin and pin in self.pin_index[country]:
            for cid in self.pin_index[country][pin][:20]:
                candidate_scores[cid] += 3
        elif pin and len(pin) >= 3 and pin[:3] in self.pin_index[country]:
            for cid in self.pin_index[country][pin][:10]:
                candidate_scores[cid] += 1
        tokens = [t for t in stripped_name.split() if len(t) >= 3 and t not in STOPWORDS]
        for tok in tokens:
            if tok in self.token_index[country]:
                matches = self.token_index[country][tok]
                if len(matches) < 5000:
                    for cid in matches[:25]:
                        candidate_scores[cid] += 2
        if metaphone_key:
            first_meta = metaphone_key.split()[0]
            if len(first_meta) >= 3 and first_meta in self.phonetic_index[country]:
                matches = self.phonetic_index[country][first_meta]
                if len(matches) < 2000:
                    for cid in matches[:15]:
                        candidate_scores[cid] += 1
        if not candidate_scores:
            return []
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sorted_candidates[:self.max_candidates]]

def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter) / union if union else 0.0

def compute_pairwise_features(s1_name, s1_meta, s1_addr, s1_pin, s1_city, s1_state,
                               cid, cand_name, cand_meta, cand_addr, cand_pin, cand_city, cand_state):
    len_a = len(s1_name)
    len_b = len(cand_name)
    max_len = max(len_a, len_b, 1)
    lev = jellyfish.levenshtein_distance(s1_name, cand_name)
    name_lev_sim = max(0.0, 1.0 - (lev / max_len))
    name_jacc = jaccard_similarity(set(s1_name.split()), set(cand_name.split()))
    name_len_ratio = min(len_a, len_b) / max_len
    s1_m = s1_meta.split()[0] if s1_meta else ""
    c_m = cand_meta.split()[0] if cand_meta else ""
    meta_sim = 1.0 if (s1_m and s1_m == c_m) else (0.5 if (s1_m and c_m and s1_m[:3] == c_m[:3]) else 0.0)
    addr_jacc = jaccard_similarity(set(s1_addr.split()), set(cand_addr.split()))
    if s1_pin and cand_pin:
        pin_score = 1.0 if s1_pin == cand_pin else (0.5 if (len(s1_pin) >= 3 and len(cand_pin) >= 3 and s1_pin[:3] == cand_pin[:3]) else -1.0)
    else:
        pin_score = 0.0
    city_score = 1.0 if (s1_city and cand_city and (s1_city == cand_city or s1_city in cand_city or cand_city in s1_city)) else (-1.0 if (s1_city and cand_city) else 0.0)
    state_score = 1.0 if (s1_state and cand_state and (s1_state == cand_state or s1_state in cand_state or cand_state in s1_state)) else (-1.0 if (s1_state and cand_state) else 0.0)
    is_s2 = 1.0 if cid.startswith("S2-") else 0.0
    return [name_lev_sim, name_jacc, name_len_ratio, meta_sim, addr_jacc, pin_score, city_score, state_score, is_s2]

def preprocess_df(df: pl.DataFrame) -> pl.DataFrame:
    entity_ids = df['entity_id'].to_list()
    raw_names = df['business_name'].to_list()
    raw_addrs = df['business_address'].to_list()
    raw_countries = df['country'].to_list()
    cleaned_names, stripped_names, metaphones = [], [], []
    countries, pins, cities, states, clean_addrs = [], [], [], [], []
    for name, addr, c in zip(raw_names, raw_addrs, raw_countries):
        canon_c = canonicalize_country(c)
        c_name, s_name, meta = clean_business_name(name)
        pin = extract_pin(addr, canon_c)
        city, state = extract_city_state(addr)
        c_addr = clean_address(addr)
        countries.append(canon_c)
        cleaned_names.append(c_name)
        stripped_names.append(s_name)
        metaphones.append(meta)
        pins.append(pin)
        cities.append(city)
        states.append(state)
        clean_addrs.append(c_addr)
    return pl.DataFrame({
        'entity_id': entity_ids,
        'country': countries,
        'cleaned_name': cleaned_names,
        'stripped_name': stripped_names,
        'metaphone': metaphones,
        'pin': pins,
        'city': cities,
        'state': states,
        'clean_addr': clean_addrs
    })

def find_dataset_dir() -> str:
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "train_source1.tsv" in files:
                return os.path.dirname(root)
    candidates = ["/kaggle/input/amazon-ml-challenge-2026/dataset", "/kaggle/input/amazon-ml-challenge-2026", "student_resource/dataset", "dataset"]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train", "train_source1.tsv")):
            return c
    raise FileNotFoundError("Dataset root not found")

def main():
    data_dir = find_dataset_dir()
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "entity_match_model.joblib")
    if not os.path.exists(model_path):
        print("Model not found at", model_path)
        sys.exit(1)
    clf = joblib.load(model_path)
    # Load candidate universe (train S2+S3)
    s2 = pl.read_csv(os.path.join(data_dir, "train", "train_source2.tsv"), separator='\t')
    s3 = pl.read_csv(os.path.join(data_dir, "train", "train_source3.tsv"), separator='\t')
    cand_raw = pl.concat([s2, s3])
    cand_df = preprocess_df(cand_raw)
    blocker = CandidateBlocker(max_candidates_per_entity=30)
    blocker.index_candidates(cand_df['entity_id'].to_list(), cand_df['country'].to_list(),
                             cand_df['stripped_name'].to_list(), cand_df['pin'].to_list(), cand_df['metaphone'].to_list())
    cand_dict = {eid: (sname, meta, caddr, pin, city, state) for eid, sname, meta, caddr, pin, city, state in zip(
        cand_df['entity_id'].to_list(), cand_df['stripped_name'].to_list(), cand_df['metaphone'].to_list(),
        cand_df['clean_addr'].to_list(), cand_df['pin'].to_list(), cand_df['city'].to_list(), cand_df['state'].to_list()
    )}
    del cand_df; gc.collect()
    # Load training S1
    s1_raw = pl.read_csv(os.path.join(data_dir, "train", "train_source1.tsv"), separator='\t')
    ordered_ids = s1_raw['entity_id'].to_list()
    s1_df = preprocess_df(s1_raw)
    del s1_raw; gc.collect()
    threshold = 0.85
    matches = {sid: [] for sid in ordered_ids}
    cand_lists = {}
    batch_feat, batch_sids, batch_cids = [], [], []
    for s1 in tqdm(s1_df.iter_rows(named=True), total=len(s1_df), desc="Training Inference"):
        sid = s1['entity_id']
        cands = blocker.find_candidates_for_entity(s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone'])
        cand_lists[sid] = cands
        for cid in cands:
            if cid in cand_dict:
                c_name, c_meta, c_addr, c_pin, c_city, c_state = cand_dict[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'], s1['pin'], s1['city'], s1['state'],
                    cid, c_name, c_meta, c_addr, c_pin, c_city, c_state
                )
                batch_feat.append(feat)
                batch_sids.append(sid)
                batch_cids.append(cid)
        if len(batch_feat) >= 60000:
            probs = clf.predict_proba(np.array(batch_feat, dtype=np.float32))[:,1]
            for sid, cid, p in zip(batch_sids, batch_cids, probs):
                if p >= threshold:
                    matches[sid].append(cid)
            batch_feat, batch_sids, batch_cids = [], [], []
    if batch_feat:
        probs = clf.predict_proba(np.array(batch_feat, dtype=np.float32))[:,1]
        for sid, cid, p in zip(batch_sids, batch_cids, probs):
            if p >= threshold:
                matches[sid].append(cid)
    # Write outputs preserving order
    cand_path = os.path.join(out_dir, "candidate_pairs_train.tsv")
    match_path = os.path.join(out_dir, "matching_results_train.tsv")
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in ordered_ids:
            f.write(f"{sid}\t{','.join(cand_lists.get(sid, []))}\n")
    with open(match_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_ids:
            f.write(f"{sid}\t{','.join(matches.get(sid, []))}\n")
    print("Inference complete. Files written:")
    print(cand_path)
    print(match_path)

if __name__ == "__main__":
    main()
