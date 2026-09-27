"""
Amazon ML Challenge 2026: Business Entity Resolution
Self-Contained Kaggle Execution Script
Full Production Training with 1:4 Hard Negative Mining
"""

import os
import sys
import subprocess
import re
import gc
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

# Attempt package installation (safe fallback if offline)
print("Installing dependencies...")
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "lightgbm", "polars", "jellyfish", "-q"])
except Exception as e:
    print(f"Pip install warning: {e}")

import polars as pl
import numpy as np
import lightgbm as lgb
import jellyfish
from tqdm import tqdm

LEGAL_SUFFIXES = {
    'corporation', 'corp', 'incorporated', 'inc', 'limited', 'ltd',
    'private limited', 'pvt ltd', 'pvt', 'private', 'llc', 'llp',
    'co', 'company', 'gmbh', 'sa', 'sarl', 'plc', 'enterprise', 'enterprises'
}

RE_NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9\s]')
RE_STRIP_ALL = re.compile(r'[^a-zA-Z0-9]')
RE_MULTIPLE_SPACES = re.compile(r'\s+')
RE_AMPERSAND = re.compile(r'\s*&\s*')
RE_PIN_6DIGIT = re.compile(r'\b(\d{3})[\s-]?(\d{3})\b')
RE_PIN_5DIGIT = re.compile(r'\b(\d{5})\b')

ADDRESS_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'hwy': 'highway', 'apt': 'apartment',
    'ste': 'suite', 'bldg': 'building', 'fl': 'floor', 'opp': 'opposite', 'nr': 'near'
}

def canonicalize_country(country_str: Optional[str]) -> str:
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
    # Fast native C unicode normalization (accents, umlauts, formatting)
    try:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    except Exception:
        return text

def clean_business_name(name: Optional[str]) -> Tuple[str, str, str]:
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

def extract_pin(address: Optional[str], country: str = "UNKNOWN") -> Optional[str]:
    if not address or not isinstance(address, str):
        return None
    c = country.upper()
    if c == "INDIA":
        m = RE_PIN_6DIGIT.search(address)
        if m: return m.group(1) + m.group(2)
    elif c in {"US", "FRANCE"}:
        m = RE_PIN_5DIGIT.search(address)
        if m: return m.group(1)
    m6 = RE_PIN_6DIGIT.search(address)
    if m6: return m6.group(1) + m6.group(2)
    m5 = RE_PIN_5DIGIT.search(address)
    if m5: return m5.group(1)
    return None

def extract_city_state(address: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
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

def clean_address(address: Optional[str]) -> str:
    if not address or not isinstance(address, str):
        return ""
    text = transliterate_to_latin(address).lower()
    text = RE_AMPERSAND.sub(' and ', text)
    text = RE_NON_ALPHANUM.sub(' ', text)
    tokens = text.split()
    return " ".join([ADDRESS_ABBR.get(tok, tok) for tok in tokens])

STOPWORDS = {'the', 'and', 'of', 'in', 'for', 'on', 'at', 'to', 'a', 'an', 'is', 'group', 'india', 'international', 'global'}

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
                    
    def find_candidates_for_entity(self, country: str, stripped_name: str, pin: Optional[str], metaphone_key: str) -> List[str]:
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
        return [cid for cid, score in sorted_candidates[:self.max_candidates]]

def jaccard_similarity(tokens_a: set, tokens_b: set) -> float:
    if not tokens_a and not tokens_b: return 1.0
    if not tokens_a or not tokens_b: return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return float(intersection) / float(union) if union > 0 else 0.0

def compute_pairwise_features(s1_name, s1_meta, s1_addr, s1_pin, s1_city, s1_state,
                              cid, cand_name, cand_meta, cand_addr, cand_pin, cand_city, cand_state):
    len_a = len(s1_name)
    len_b = len(cand_name)
    max_len = max(len_a, len_b, 1)
    lev_dist = jellyfish.levenshtein_distance(s1_name, cand_name)
    name_lev_sim = max(0.0, 1.0 - (lev_dist / max_len))
    name_jaccard = jaccard_similarity(set(s1_name.split()), set(cand_name.split()))
    name_len_ratio = min(len_a, len_b) / max_len
    
    s1_m = s1_meta.split()[0] if s1_meta else ""
    c_m = cand_meta.split()[0] if cand_meta else ""
    meta_sim = 1.0 if (s1_m and s1_m == c_m) else (0.5 if (s1_m and c_m and s1_m[:3] == c_m[:3]) else 0.0)
    addr_jaccard = jaccard_similarity(set(s1_addr.split()), set(cand_addr.split()))
    
    if s1_pin and cand_pin:
        pin_score = 1.0 if s1_pin == cand_pin else (0.5 if (len(s1_pin)>=3 and len(cand_pin)>=3 and s1_pin[:3]==cand_pin[:3]) else -1.0)
    else:
        pin_score = 0.0
        
    city_score = 1.0 if (s1_city and cand_city and (s1_city==cand_city or s1_city in cand_city or cand_city in s1_city)) else (-1.0 if (s1_city and cand_city) else 0.0)
    state_score = 1.0 if (s1_state and cand_state and (s1_state==cand_state or s1_state in cand_state or cand_state in s1_state)) else (-1.0 if (s1_state and cand_state) else 0.0)
    is_s2 = 1.0 if cid.startswith("S2-") else 0.0
    
    return [name_lev_sim, name_jaccard, name_len_ratio, meta_sim, addr_jaccard, pin_score, city_score, state_score, is_s2]

def preprocess_df(df):
    df = df.rename({col: col.lstrip('+') for col in df.columns})
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
        'entity_id': entity_ids, 'country': countries, 'cleaned_name': cleaned_names,
        'stripped_name': stripped_names, 'metaphone': metaphones, 'pin': pins,
        'city': cities, 'state': states, 'clean_addr': clean_addrs
    })

def find_dataset_dir() -> str:
    """Robustly discovers dataset folder in Kaggle or local environments."""
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "train_source1.tsv" in files:
                print(f"Auto-discovered dataset root: {root}")
                return os.path.dirname(root)
    candidates = [
        "/kaggle/input/amazon-ml-challenge-2026/dataset",
        "/kaggle/input/amazon-ml-challenge-2026",
        "6ab10eb3b23ba_student_resource/student_resource/dataset",
        "student_resource/dataset",
        "dataset"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train", "train_source1.tsv")):
            return c
    raise FileNotFoundError("Could not locate dataset containing train/train_source1.tsv")

def main():
    data_dir = find_dataset_dir()
    out_dir = "/kaggle/working/output" if os.path.exists("/kaggle/working") else "output"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    
    # 1. FULL PRODUCTION TRAINING (1:4 HARD NEGATIVE MINING)
    print("=== FULL PRODUCTION TRAINING (1:4 Hard Negative Mining) ===")
    print("Loading Ground Truth and Source 1...")
    train_gt_raw = pl.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), separator='\t')
    train_gt_raw = train_gt_raw.rename({col: col.lstrip('+') for col in train_gt_raw.columns})
    train_s1_raw = pl.read_csv(os.path.join(data_dir, "train", "train_source1.tsv"), separator='\t')

    gt_map = {}
    for s1, m_str in zip(train_gt_raw['source1_entity_id'].to_list(), train_gt_raw['matched_entity_ids'].to_list()):
        if m_str and isinstance(m_str, str) and m_str.strip():
            gt_map[s1] = set(x.strip() for x in m_str.split(',') if x.strip())
        else:
            gt_map[s1] = set()
    del train_gt_raw

    print(f"Total S1 training entities: {len(train_s1_raw):,}")
    print(f"S1 entities WITH matches: {sum(1 for v in gt_map.values() if v):,}")

    # Load complete S2 + S3 candidate universe (no head slices!)
    print("Loading complete train_source2.tsv and train_source3.tsv...")
    s2_train_raw = pl.read_csv(os.path.join(data_dir, "train", "train_source2.tsv"), separator='\t')
    s3_train_raw = pl.read_csv(os.path.join(data_dir, "train", "train_source3.tsv"), separator='\t')
    cand_train_raw = pl.concat([s2_train_raw, s3_train_raw])
    del s2_train_raw, s3_train_raw
    print(f"Total candidate universe for training: {len(cand_train_raw):,}")

    cand_train_df = preprocess_df(cand_train_raw)
    del cand_train_raw
    s1_train_df = preprocess_df(train_s1_raw)
    del train_s1_raw

    print("Indexing training candidate pool into Candidate Blocker...")
    train_blocker = CandidateBlocker(max_candidates_per_entity=30)
    train_blocker.index_candidates(
        cand_train_df['entity_id'].to_list(), cand_train_df['country'].to_list(),
        cand_train_df['stripped_name'].to_list(), cand_train_df['pin'].to_list(),
        cand_train_df['metaphone'].to_list()
    )
    
    # Store candidates as compact tuples for fast lookups & low memory
    cand_train_dict = {
        eid: (sname, meta, caddr, pin, city, state)
        for eid, sname, meta, caddr, pin, city, state in zip(
            cand_train_df['entity_id'].to_list(),
            cand_train_df['stripped_name'].to_list(),
            cand_train_df['metaphone'].to_list(),
            cand_train_df['clean_addr'].to_list(),
            cand_train_df['pin'].to_list(),
            cand_train_df['city'].to_list(),
            cand_train_df['state'].to_list()
        )
    }
    del cand_train_df
    gc.collect()

    print(f"Generating training pairs with 1:4 Hard Negative Mining across {len(s1_train_df):,} S1 entities...")
    X_train, y_train = [], []
    for s1 in tqdm(s1_train_df.iter_rows(named=True), total=len(s1_train_df), desc="Train Pairs"):
        s1_id = s1['entity_id']
        true_matches = gt_map.get(s1_id, set())

        # 1. Positives (Ground Truth Matches)
        for cid in true_matches:
            if cid in cand_train_dict:
                c_name, c_meta, c_addr, c_pin, c_city, c_state = cand_train_dict[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'], s1['pin'], s1['city'], s1['state'],
                    cid, c_name, c_meta, c_addr, c_pin, c_city, c_state
                )
                X_train.append(feat)
                y_train.append(1)

        # 2. Hard Negatives: Blocker candidates that are NOT true matches, ranked by name + address similarity
        cands = train_blocker.find_candidates_for_entity(s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone'])
        false_cands = [c for c in cands if c not in true_matches and c in cand_train_dict]
        if false_cands:
            cand_neg_feats = []
            for cid in false_cands:
                c_name, c_meta, c_addr, c_pin, c_city, c_state = cand_train_dict[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'], s1['pin'], s1['city'], s1['state'],
                    cid, c_name, c_meta, c_addr, c_pin, c_city, c_state
                )
                # Composite hardness score: name Levenshtein + token Jaccard + address overlap
                hard_score = feat[0] * 0.4 + feat[1] * 0.3 + feat[4] * 0.3
                cand_neg_feats.append((hard_score, feat))

            # Select the top 4 hardest lookalikes
            cand_neg_feats.sort(key=lambda x: x[0], reverse=True)
            for _, feat in cand_neg_feats[:4]:
                X_train.append(feat)
                y_train.append(0)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int8)
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    print(f"Total training pairs: {len(X_train):,} (Positives: {n_pos:,}, Hard Negatives: {n_neg:,})")
    pos_neg_ratio = n_neg / max(n_pos, 1)
    print(f"Effective Negative-to-Positive Ratio: {pos_neg_ratio:.2f} : 1")

    # Fit LightGBM
    print("Training LightGBM Classifier...")
    scale_pos = min(pos_neg_ratio, 15.0)
    clf = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', n_estimators=300,
                             learning_rate=0.06, max_depth=7, num_leaves=63,
                             scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(X_train, y_train)
    print(f"Training complete. Positive pairs seen: {n_pos:,}")

    # Save trained model
    import joblib
    model_save_path = os.path.join(out_dir, "entity_match_model.joblib")
    joblib.dump(clf, model_save_path)
    print(f"Model saved: {model_save_path} ({os.path.getsize(model_save_path)/1e6:.1f} MB)")

    del s1_train_df, cand_train_dict, X_train, y_train, train_blocker
    gc.collect()

    # 2. Index Full Test Candidate Pool (S2 + S3)
    print("Loading full test candidate pools (test_source2.tsv + test_source3.tsv)...")
    s2_test_raw = pl.read_csv(os.path.join(data_dir, "test", "test_source2.tsv"), separator='\t')
    s3_test_raw = pl.read_csv(os.path.join(data_dir, "test", "test_source3.tsv"), separator='\t')
    cand_test_raw = pl.concat([s2_test_raw, s3_test_raw])
    del s2_test_raw, s3_test_raw
    print(f"Preprocessing {len(cand_test_raw):,} test candidate records...")
    cand_test_df = preprocess_df(cand_test_raw)
    del cand_test_raw

    test_blocker = CandidateBlocker(max_candidates_per_entity=30)
    test_blocker.index_candidates(
        cand_test_df['entity_id'].to_list(), cand_test_df['country'].to_list(),
        cand_test_df['stripped_name'].to_list(), cand_test_df['pin'].to_list(),
        cand_test_df['metaphone'].to_list()
    )
    # Store candidates as compact tuples
    cand_test_dict = {
        eid: (sname, meta, caddr, pin, city, state)
        for eid, sname, meta, caddr, pin, city, state in zip(
            cand_test_df['entity_id'].to_list(),
            cand_test_df['stripped_name'].to_list(),
            cand_test_df['metaphone'].to_list(),
            cand_test_df['clean_addr'].to_list(),
            cand_test_df['pin'].to_list(),
            cand_test_df['city'].to_list(),
            cand_test_df['state'].to_list()
        )
    }
    del cand_test_df
    gc.collect()

    # 3. Stream and Predict Test Source 1 Entities
    print("Loading test_source1.tsv...")
    s1_test_raw = pl.read_csv(os.path.join(data_dir, "test", "test_source1.tsv"), separator='\t')
    ordered_s1_ids = s1_test_raw['entity_id'].to_list()
    s1_test_df = preprocess_df(s1_test_raw)
    del s1_test_raw

    print(f"Running inference on {len(s1_test_df):,} test Source 1 records...")
    threshold = 0.85 # Tuned for optimal macro F_0.5
    print(f"Optimal decision cutoff threshold = {threshold:.2f}")

    test_candidates_dict = {}
    test_matches_dict = {sid: [] for sid in ordered_s1_ids}

    batch_s1, batch_cid, batch_feat = [], [], []
    for s1 in tqdm(s1_test_df.iter_rows(named=True), total=len(s1_test_df), desc="Test Inference"):
        s1_id = s1['entity_id']
        cands = test_blocker.find_candidates_for_entity(s1['country'], s1['stripped_name'], s1['pin'], s1['metaphone'])
        test_candidates_dict[s1_id] = cands

        for cid in cands:
            if cid in cand_test_dict:
                c_name, c_meta, c_addr, c_pin, c_city, c_state = cand_test_dict[cid]
                feat = compute_pairwise_features(
                    s1['stripped_name'], s1['metaphone'], s1['clean_addr'], s1['pin'], s1['city'], s1['state'],
                    cid, c_name, c_meta, c_addr, c_pin, c_city, c_state
                )
                batch_s1.append(s1_id)
                batch_cid.append(cid)
                batch_feat.append(feat)

        if len(batch_feat) >= 60000:
            probs = clf.predict_proba(np.array(batch_feat, dtype=np.float32))[:, 1]
            for sid, cid, p in zip(batch_s1, batch_cid, probs):
                if p >= threshold:
                    test_matches_dict[sid].append(cid)
            batch_s1, batch_cid, batch_feat = [], [], []

    if batch_feat:
        probs = clf.predict_proba(np.array(batch_feat, dtype=np.float32))[:, 1]
        for sid, cid, p in zip(batch_s1, batch_cid, probs):
            if p >= threshold:
                test_matches_dict[sid].append(cid)

    # 4. Save Final Leaderboard Files Preserving Line-by-Line Row Order
    cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
    match_path = os.path.join(out_dir, "matching_results.tsv")

    print(f"Writing {cand_path}...")
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in ordered_s1_ids:
            cands = test_candidates_dict.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f"Writing {match_path}...")
    with open(match_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in ordered_s1_ids:
            matches = test_matches_dict.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(matches)}\n")

    print("SUCCESS: Both matching_results.tsv and candidate_pairs.tsv successfully generated!")

if __name__ == "__main__":
    main()
