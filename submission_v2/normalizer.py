"""
Production-grade normalizer with enhanced address token extraction.
Handles US, India (Devanagari/Tamil transliteration), and France (accents/diacritics).
"""

import re
import unicodedata
from typing import List, Optional, Set, Tuple


# ─── Unicode / Diacritics ────────────────────────────────────────────────────

def strip_accents(text: str) -> str:
    """NFKD-normalize then drop combining marks (é→e, ç→c, ñ→n, ā→a …)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_unicode(text: str) -> str:
    """Lower-case + strip accents + collapse whitespace."""
    if not text or text in ("None", "null", "NaN", "nan"):
        return ""
    t = strip_accents(text).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ─── Legal Suffix Stripping ──────────────────────────────────────────────────

# Includes US/India/France legal forms and common abbreviations
_LEGAL_RE = re.compile(
    r"\b("
    r"private\s+limited|pvt\.?\s*ltd\.?|pvt\s+ltd|pvt\s+limited"
    r"|limited\s+liability\s+company|llc|inc\.?|incorporated"
    r"|corp\.?|corporation|ltd\.?|limited|llp|gmbh|co\.?|company"
    r"|sarl|s\.a\.r\.l\.?|sasu|s\.a\.s\.u\.?|sas|s\.a\.s\.?"
    r"|eurl|e\.u\.r\.l\.?|sci|s\.c\.i\.?|sa\b|s\.a\.?|gie|scop"
    r"|p\.?\s*ltd\.?|pvt"
    r")\b",
    re.IGNORECASE,
)

def strip_legal(text: str) -> str:
    cleaned = _LEGAL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


# ─── Street Abbreviations ────────────────────────────────────────────────────

_STREET_ABB = {
    "st": "street", "str": "street",
    "rd": "road", "rd.": "road",
    "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bvd": "boulevard", "bd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "pkwy": "parkway", "hwy": "highway", "sq": "square",
    "cir": "circle", "apt": "apartment", "ste": "suite",
    "bldg": "building", "fl": "floor",
    # French
    "r": "rue", "r.": "rue", "all": "allee",
}

# Tokens that are not useful for blocking (too generic)
_STOP_ADDR = {
    "road", "street", "avenue", "lane", "drive", "near", "behind",
    "opp", "opposite", "colony", "main", "cross", "floor", "first",
    "second", "third", "unit", "building", "india", "state", "us",
    "usa", "france", "rue", "de", "la", "le", "des", "du", "en",
    "area", "sector", "block", "plot", "house", "flat", "shop",
    "no", "null", "none", "nagar", "marg", "gali",
}

_NUM_RE = re.compile(r"\b(\d+)\b")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


# ─── Public API ─────────────────────────────────────────────────────────────

def clean_name(raw: Optional[str]) -> Tuple[str, str]:
    """
    Returns (core_name, normalized_name).
    core_name: legal suffixes removed (for similarity comparison).
    normalized_name: cleaned but legal suffixes kept.
    """
    if not raw or raw in ("None", "null", "NaN", "nan"):
        return "", ""
    norm = normalize_unicode(raw)
    clean = _PUNCT_RE.sub(" ", norm)
    clean = _WS_RE.sub(" ", clean).strip()
    core = strip_legal(clean)
    if not core:
        core = clean
    return core, clean


def clean_address(raw: Optional[str]) -> Tuple[str, List[str], Set[str]]:
    """
    Returns (cleaned_str, sorted_distinctive_tokens, normalized_numeric_tokens).
    - sorted_distinctive_tokens: stop-words and digits removed, sorted.
    - normalized_numeric_tokens: leading-zero-stripped numeric strings.
    """
    if not raw or raw in ("None", "null", "NaN", "nan"):
        return "", [], set()

    norm = normalize_unicode(raw)

    # Extract numeric tokens before removing punctuation
    raw_nums = set(_NUM_RE.findall(norm))
    # Strip leading zeros (AF-0684 → 684 == Af-684)
    norm_nums: Set[str] = set()
    for n in raw_nums:
        stripped = n.lstrip("0") or "0"
        norm_nums.add(stripped)

    clean = _PUNCT_RE.sub(" ", norm)
    tokens = _WS_RE.sub(" ", clean).strip().split()

    # Expand abbreviations
    expanded = []
    for t in tokens:
        expanded.append(_STREET_ABB.get(t, t))

    # Distinctive tokens: non-digit, non-stopword, len >= 4
    distinctive = sorted(
        {t for t in expanded if t not in _STOP_ADDR and not t.isdigit() and len(t) >= 4}
    )

    cleaned_str = " ".join(expanded)
    return cleaned_str, distinctive, norm_nums


def name_bigrams(core_name: str, n: int = 3) -> Set[str]:
    """Character n-grams from a cleaned name for fuzzy blocking."""
    s = core_name.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def extract_blocking_keys(
    country: str,
    core_name: str,
    distinctive_addr: List[str],
    norm_nums: Set[str],
) -> List[str]:
    """
    Generate all candidate blocking keys for an entity.
    Each key is prefixed so different key types never collide.

    Key types:
      nw1/nw2/nw3  - first 3 words of core business name
      atk          - distinctive address token (up to 4)
      num          - normalised numeric token (house no, zip, pin)
      na           - name_word1 + addr_token1 combo
      nt           - numeric + addr_token combo (covers transliteration cases)
      aa           - pair of address tokens (covers reordering/partial)
    """
    keys: List[str] = []
    c = country.strip() if country else "?"
    words = core_name.split()

    # ── Name-prefix keys (first 3 words) ──────────────────────────────────────
    for i, w in enumerate(words[:3]):
        if len(w) >= 3:
            keys.append(f"{c}|nw{i+1}|{w[:5]}")

    # ── Address distinctive token keys (up to 4) ───────────────────────────────
    for tok in distinctive_addr[:4]:
        keys.append(f"{c}|atk|{tok[:6]}")

    # ── Normalised numeric keys ────────────────────────────────────────────────
    for num in norm_nums:
        if 3 <= len(num) <= 7:
            keys.append(f"{c}|num|{num}")

    # ── Name word 1 + Address token 1 combo ───────────────────────────────────
    if words and distinctive_addr:
        keys.append(f"{c}|na|{words[0][:4]}_{distinctive_addr[0][:5]}")

    # ── Numeric + Address token combo (critical for transliteration cases) ─────
    # "Ss Food" ↔ "एसएस फूड" share address nums (684) + addr token (nandg)
    for num in norm_nums:
        if 3 <= len(num) <= 7:
            for tok in distinctive_addr[:2]:
                keys.append(f"{c}|nt|{num}_{tok[:5]}")

    # ── Pairs of address tokens (catches partial/reordered addresses) ──────────
    if len(distinctive_addr) >= 2:
        keys.append(f"{c}|aa|{distinctive_addr[0][:5]}_{distinctive_addr[1][:5]}")

    # ── Triple address token (city + locality + building/no) ──────────────────
    if len(distinctive_addr) >= 3:
        keys.append(f"{c}|aaa|{distinctive_addr[0][:4]}_{distinctive_addr[2][:4]}")

    return keys

