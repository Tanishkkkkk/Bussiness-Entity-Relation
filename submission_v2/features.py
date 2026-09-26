"""
Phase 3 & 5 — High-Speed Pairwise Feature Extraction.

Features:
  [0-4]   Name similarity   (jaro_winkler, ngram_3, ngram_2, prefix_match, token_sort_ratio)
  [5-8]   Address similarity (token_jaccard, num_match, distinct_overlap, addr_prefix)
  [9-11]  Structural         (country_match, legal_form_match, name_len_ratio)
  [12-13] Combined           (name*addr, max_single_signal)
  [14]    Blocking score     (normalized score from blocker)
  [15]    exact_core_match   (1.0 if normalized core names identical)
  [16]    token_recall_a     (fraction of A tokens present in B)
  [17]    token_recall_b     (fraction of B tokens present in A)
  [18]    min_token_recall   (min of recall_a and recall_b)
  [19]    name_contains      (1 if one name is substring of the other)
  [20]    addr_num_exact     (1 if any shared street number token exists)

Total: 21 features. All values in [0, 1].
"""

import re
import math
from typing import Dict, List, Optional, Set, Tuple

from normalizer import clean_name, clean_address


# ─── Jaro-Winkler ────────────────────────────────────────────────────────────

def _jaro(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0:
        return 0.0
    match_dist = max(l1, l2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s1_matches = [False] * l1
    s2_matches = [False] * l2
    matches = 0
    transpositions = 0
    for i in range(l1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, l2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(l1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    return (matches / l1 + matches / l2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler similarity, p=0.1 (standard prefix weight)."""
    j = _jaro(s1, s2)
    prefix = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a == b:
            prefix += 1
        else:
            break
    return j + prefix * p * (1 - j)


# ─── Character n-gram overlap ─────────────────────────────────────────────────

def _char_ngrams(text: str, n: int) -> Set[str]:
    s = text.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """Dice coefficient over character n-grams."""
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    return 2 * inter / (len(ga) + len(gb))


# ─── Token sort ratio ─────────────────────────────────────────────────────────

def token_sort_ratio(a: str, b: str) -> float:
    """Sort tokens alphabetically then compute Jaro-Winkler."""
    ta = " ".join(sorted(a.split()))
    tb = " ".join(sorted(b.split()))
    return jaro_winkler(ta, tb)


# ─── Prefix match ─────────────────────────────────────────────────────────────

def prefix_match(a: str, b: str, length: int = 5) -> float:
    """Exact prefix match of first `length` characters."""
    pa, pb = a[:length], b[:length]
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0
    return 1.0 if pa == pb else 0.0


# ─── Token Jaccard ────────────────────────────────────────────────────────────

def token_jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ─── Numeric match ────────────────────────────────────────────────────────────

def numeric_match(na: Set[str], nb: Set[str]) -> float:
    """Fraction of numeric tokens shared (normalized)."""
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    shared = len(na & nb)
    return shared / max(len(na), len(nb))


# ─── Address distinctive overlap ──────────────────────────────────────────────

def distinct_addr_overlap(da: List[str], db: List[str]) -> float:
    sa, sb = set(da), set(db)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ─── Legal form extraction ────────────────────────────────────────────────────

_LEGAL_TAGS = {
    "llc", "inc", "corp", "ltd", "limited", "llp", "pvt",
    "gmbh", "sa", "sas", "sarl", "sasu", "eurl", "gie", "scop", "co",
}

def _get_legal_form(name: str) -> str:
    """Returns the detected legal form tag or empty string."""
    for tok in name.split():
        t = tok.strip(".,")
        if t in _LEGAL_TAGS:
            return t
    return ""


# ─── New discriminative features ─────────────────────────────────────────────

def token_recall(a: str, b: str) -> float:
    """Fraction of a's tokens that appear in b's token set."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta:
        return 1.0 if not tb else 0.0
    return len(ta & tb) / len(ta)


def name_contains_check(a: str, b: str) -> float:
    """1.0 if one clean name is a substring of the other."""
    a_ns = a.replace(" ", "")
    b_ns = b.replace(" ", "")
    if not a_ns or not b_ns:
        return 0.0
    return 1.0 if (a_ns in b_ns or b_ns in a_ns) else 0.0


def addr_num_exact(na: Set[str], nb: Set[str]) -> float:
    """1.0 if at least one numeric token matches exactly (street number)."""
    if not na or not nb:
        return 0.0
    return 1.0 if na & nb else 0.0


# ─── Precomputed Feature Extractor (Ultra-Fast) ──────────────────────────────

def extract_features_precomputed(
    s1_core: str,
    s1_full: str,
    s1_addr_str: str,
    s1_dist: List[str],
    s1_nums: Set[str],
    s1_ctry: str,
    cd_core: str,
    cd_full: str,
    cd_addr_str: str,
    cd_dist: List[str],
    cd_nums: Set[str],
    cd_ctry: str,
    blocking_score: float = 0.0,
    max_blocking_score: float = 20.0,
) -> List[float]:
    """
    Extract 21 pairwise features directly from pre-normalized fields.
    Zero string parsing overhead.
    """
    # Name features [0-4]
    f0 = jaro_winkler(s1_core, cd_core)
    f1 = ngram_overlap(s1_core, cd_core, n=3)
    f2 = ngram_overlap(s1_core, cd_core, n=2)
    f3 = prefix_match(s1_core, cd_core, length=5)
    f4 = token_sort_ratio(s1_core, cd_core)

    # Address features [5-8]
    f5 = token_jaccard(s1_addr_str, cd_addr_str)
    f6 = numeric_match(s1_nums, cd_nums)
    f7 = distinct_addr_overlap(s1_dist, cd_dist)
    f8 = prefix_match(s1_addr_str, cd_addr_str, length=8)

    # Structural features [9-11]
    f9  = 1.0 if s1_ctry == cd_ctry else 0.0
    lf1 = _get_legal_form(s1_full)
    lf2 = _get_legal_form(cd_full)
    f10 = 1.0 if lf1 and lf2 and lf1 == lf2 else (0.5 if lf1 == lf2 == "" else 0.0)
    l1  = len(s1_core.replace(" ", "")) + 1
    l2  = len(cd_core.replace(" ", "")) + 1
    f11 = min(l1, l2) / max(l1, l2)

    # Combined signals [12-13]
    f12 = f0 * f5
    f13 = max(f0, f1, f5, f6, f7)

    # Normalized blocking score [14]
    f14 = min(blocking_score / max_blocking_score, 1.0) if max_blocking_score > 0 else 0.0

    # New discriminative features [15-20]
    f15 = 1.0 if s1_core == cd_core and s1_core != "" else 0.0  # exact core name match
    f16 = token_recall(s1_core, cd_core)    # recall: A tokens in B
    f17 = token_recall(cd_core, s1_core)    # recall: B tokens in A
    f18 = min(f16, f17)                     # min token recall (both directions)
    f19 = name_contains_check(s1_core, cd_core)  # substring containment
    f20 = addr_num_exact(s1_nums, cd_nums)        # exact street number match

    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14,
            f15, f16, f17, f18, f19, f20]


def extract_features(
    s1: Dict,
    cand: Dict,
    blocking_score: float = 0.0,
    max_blocking_score: float = 20.0,
) -> List[float]:
    """Compatibility wrapper for dictionary inputs."""
    s1_core, s1_full = clean_name(s1.get("business_name"))
    cd_core, cd_full = clean_name(cand.get("business_name"))
    s1_addr_str, s1_dist, s1_nums = clean_address(s1.get("business_address"))
    cd_addr_str, cd_dist, cd_nums = clean_address(cand.get("business_address"))
    s1_ctry = (s1.get("country") or "").strip().lower()
    cd_ctry = (cand.get("country") or "").strip().lower()

    return extract_features_precomputed(
        s1_core, s1_full, s1_addr_str, s1_dist, s1_nums, s1_ctry,
        cd_core, cd_full, cd_addr_str, cd_dist, cd_nums, cd_ctry,
        blocking_score, max_blocking_score
    )


FEATURE_NAMES = [
    "jaro_winkler",
    "ngram3_overlap",
    "ngram2_overlap",
    "prefix5_match",
    "token_sort_ratio",
    "addr_token_jaccard",
    "numeric_match",
    "distinct_addr_ol",
    "addr_prefix8",
    "country_match",
    "legal_form_match",
    "name_len_ratio",
    "name_x_addr",
    "max_signal",
    "blocking_score_n",
    # New features (v2)
    "exact_core_match",
    "token_recall_a",
    "token_recall_b",
    "min_token_recall",
    "name_contains",
    "addr_num_exact",
]
