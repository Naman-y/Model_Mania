import re
import jellyfish
from typing import Optional, Tuple, Set, List

try:
    from aksharamukha import transliterate
    HAS_AKSHARAMUKHA = True
except ImportError:
    HAS_AKSHARAMUKHA = False

# Legal suffixes to strip from business names for cleaner matching
LEGAL_SUFFIXES = {
    'corporation', 'corp', 'incorporated', 'inc', 'limited', 'ltd',
    'private limited', 'pvt ltd', 'pvt', 'private', 'llc', 'llp',
    'co', 'company', 'gmbh', 'sa', 'sarl', 'plc',
    'enterprise', 'enterprises'
}

# Regex patterns
RE_NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9\s]')
RE_STRIP_ALL = re.compile(r'[^a-zA-Z0-9]')
RE_MULTIPLE_SPACES = re.compile(r'\s+')
RE_AMPERSAND = re.compile(r'\s*&\s*')

# Postal code regex:
# 1) 6-digit Indian PIN (e.g. 560001, 560 001, 560-001) -> capture as 6 digits
RE_PIN_6DIGIT = re.compile(r'\b(\d{3})[\s-]?(\d{3})\b')
# 2) 5-digit US / France ZIP/Postal code (e.g. 90210, 75008)
RE_PIN_5DIGIT = re.compile(r'\b(\d{5})\b')

# Standard abbreviations for address normalization
ADDRESS_ABBR = {
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'blvd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'hwy': 'highway',
    'apt': 'apartment',
    'ste': 'suite',
    'bldg': 'building',
    'fl': 'floor',
    'opp': 'opposite',
    'nr': 'near'
}

def canonicalize_country(country_str: Optional[str]) -> str:
    """Canonicalize country strings to prevent silent blocking splits."""
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
    """Convert native script text (Devanagari, Tamil, Telugu, etc.) to Latin using Aksharamukha."""
    if not text:
        return ""
    # Quick check if non-ASCII characters exist
    if all(ord(ch) < 128 for ch in text):
        return text
    if HAS_AKSHARAMUKHA:
        try:
            # Aksharamukha autodetects the source script when given 'autodetect'
            return transliterate.process('autodetect', 'RomanReadable', text)
        except Exception:
            try:
                return transliterate.process('autodetect', 'ISO', text)
            except Exception:
                return text
    return text

def clean_business_name(name: Optional[str]) -> Tuple[str, str, str]:
    """
    Cleans business name:
    Returns (cleaned_name, stripped_name, metaphone_key).
    """
    if not name or not isinstance(name, str):
        return ("", "", "")
    
    # 1. Transliterate if non-ASCII
    text = transliterate_to_latin(name)
    
    # 2. Lowercase and replace & with and
    text = text.lower()
    text = RE_AMPERSAND.sub(' and ', text)
    
    # 3. Strip punctuation
    text = RE_NON_ALPHANUM.sub(' ', text)
    tokens = [t for t in RE_MULTIPLE_SPACES.sub(' ', text).strip().split() if t]
    cleaned_name = " ".join(tokens)
    
    # 4. Strip legal suffixes
    filtered_tokens = []
    # Check multi-word suffixes first
    joined = " " + cleaned_name + " "
    for suffix in ['private limited', 'pvt ltd']:
        if joined.endswith(f" {suffix} "):
            joined = joined[:-len(suffix)-2]
    
    tokens = joined.strip().split()
    for tok in tokens:
        if tok not in LEGAL_SUFFIXES:
            filtered_tokens.append(tok)
            
    stripped_name = " ".join(filtered_tokens) if filtered_tokens else cleaned_name
    
    # 5. Metaphone phonetic skeleton
    # Jellyfish metaphone on the first 2-3 key tokens
    metaphone_tokens = [jellyfish.metaphone(tok) for tok in (filtered_tokens[:3] if filtered_tokens else tokens[:3])]
    metaphone_key = " ".join([m for m in metaphone_tokens if m])
    
    return cleaned_name, stripped_name, metaphone_key

def extract_pin(address: Optional[str], country: str = "UNKNOWN") -> Optional[str]:
    """
    Extract postal code using tolerant regex.
    Prioritizes 6-digit PIN for India, 5-digit for US/France.
    """
    if not address or not isinstance(address, str):
        return None
    
    c = country.upper()
    if c == "INDIA":
        # Check 6-digit first
        m = RE_PIN_6DIGIT.search(address)
        if m:
            return m.group(1) + m.group(2)
    elif c in {"US", "FRANCE"}:
        m = RE_PIN_5DIGIT.search(address)
        if m:
            return m.group(1)
            
    # Generic fallback
    m6 = RE_PIN_6DIGIT.search(address)
    if m6:
        return m6.group(1) + m6.group(2)
    m5 = RE_PIN_5DIGIT.search(address)
    if m5:
        return m5.group(1)
        
    return None

def extract_city_state(address: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Simple, bounded last-two-segments rule per Implementation Plan Section 4:
    Accept last 2 comma-separated segments as (city, state) ONLY if neither contains a digit
    and neither is a literal substring of the other. Else return (None, None).
    """
    if not address or not isinstance(address, str):
        return (None, None)
        
    parts = [p.strip().lower() for p in address.split(',') if p.strip()]
    if len(parts) < 2:
        return (None, None)
        
    candidate_city = RE_NON_ALPHANUM.sub(' ', parts[-2]).strip()
    candidate_state = RE_NON_ALPHANUM.sub(' ', parts[-1]).strip()
    
    # Rule 1: Neither contains digits
    if any(ch.isdigit() for ch in candidate_city) or any(ch.isdigit() for ch in candidate_state):
        return (None, None)
        
    # Rule 2: Non-empty after clean
    if not candidate_city or not candidate_state:
        return (None, None)
        
    # Rule 3: Neither is a substring of the other
    if candidate_city in candidate_state or candidate_state in candidate_city:
        return (None, None)
        
    return (candidate_city, candidate_state)

def clean_address(address: Optional[str]) -> str:
    """Normalize full address string with abbreviation expansion."""
    if not address or not isinstance(address, str):
        return ""
    text = transliterate_to_latin(address).lower()
    text = RE_AMPERSAND.sub(' and ', text)
    text = RE_NON_ALPHANUM.sub(' ', text)
    tokens = text.split()
    expanded = [ADDRESS_ABBR.get(tok, tok) for tok in tokens]
    return " ".join(expanded)
