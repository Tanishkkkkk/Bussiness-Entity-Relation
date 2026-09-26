"""
Phase 3 — Pairwise Dataset Builder.

Constructs a balanced, high-quality pairwise training dataset:
  1. Samples 30,000 S1 records from train_ground_truth.tsv
  2. Extracts all true positive match targets
  3. Pools targets + 100k distractor records (S2 + S3)
  4. Runs blocking to find hard negative candidates
  5. Computes 15 pairwise similarity features
  6. Exports dataset/processed/train_pairs.parquet

Memory safe, high-speed execution (~1 minute total).
"""

import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import polars as pl

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
sys.stdout.reconfigure(encoding="utf-8")

from blocker import BlockingIndex, build_index_from_df
from features import extract_features, FEATURE_NAMES
from normalizer import clean_name, clean_address

OUTPUT_PATH = "dataset/processed/train_pairs.parquet"
os.makedirs("dataset/processed", exist_ok=True)

N_SAMPLE_S1 = 100_000   # increased from 30K for better coverage
N_DISTRACTORS = 100_000  # increased from 50K for harder negatives
TOP_K = 20              # more candidates per entity for harder negatives


def build_dataset():
    BASE = "dataset/train/"

    print("=" * 60)
    print("  PHASE 3: PAIRWISE DATASET CONSTRUCTION (FAST & LEAN)")
    print("=" * 60)

    # ── 1. Load sample ground truth (country-stratified) ─────────────────────
    print(f"\n[1/5] Loading ground truth (stratified sample of {N_SAMPLE_S1:,}) ...")
    # Load full GT then stratify by country
    gt_all = pl.read_csv(BASE + "train_ground_truth.tsv", separator="\t")
    s1_meta = pl.scan_csv(BASE + "train_source1.tsv", separator="\t").select(["entity_id", "country"]).collect()
    gt_all = gt_all.join(s1_meta, left_on="source1_entity_id", right_on="entity_id", how="left")

    # Sample proportionally per country
    countries = gt_all["country"].unique().to_list()
    sampled = []
    for ctry in countries:
        sub = gt_all.filter(pl.col("country") == ctry)
        n = min(len(sub), int(N_SAMPLE_S1 * len(sub) / len(gt_all)) + 1)
        sampled.append(sub.sample(n=n, seed=42))
    gt_df = pl.concat(sampled).head(N_SAMPLE_S1)
    print(f"  Sampled {len(gt_df):,} entities | Country distribution:")
    print(gt_df["country"].value_counts().sort("count", descending=True))

    gt: Dict[str, Set[str]] = {}
    s1_ids: List[str] = []
    target_ids: Set[str] = set()

    for r in gt_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        s1_ids.append(sid)
        m = r.get("matched_entity_ids") or ""
        matches = set(m.split(",")) if m.strip() else set()
        gt[sid] = matches
        target_ids.update(matches)

    n_positives_total = sum(len(v) for v in gt.values())
    n_no_match = sum(1 for v in gt.values() if not v)
    print(f"  {len(s1_ids):,} S1 entities | {n_positives_total:,} true targets | {n_no_match:,} no-match")

    # ── 2. Load S1 records ───────────────────────────────────────────────────
    print("\n[2/5] Loading matching S1 records …")
    s1_df = (
        pl.scan_csv(BASE + "train_source1.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(s1_ids))
        .collect()
    )
    s1_dict: Dict[str, dict] = {r["entity_id"]: r for r in s1_df.to_dicts()}
    print(f"  Loaded {len(s1_dict):,} S1 records")

    # ── 3. Load Candidate Pool (Targets + Distractors) ────────────────────────
    print(f"\n[3/5] Loading candidate pool ({len(target_ids):,} targets + {N_DISTRACTORS*2:,} distractors) ...")
    s2_tgt_ids = [x for x in target_ids if x.startswith("S2-")]
    s3_tgt_ids = [x for x in target_ids if x.startswith("S3-")]

    s2_targets = (
        pl.scan_csv(BASE + "train_source2.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(s2_tgt_ids))
        .collect()
    )
    s3_targets = (
        pl.scan_csv(BASE + "train_source3.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(s3_tgt_ids))
        .collect()
    )

    # Sample distractors from ENTIRE S2/S3 (not just first N rows) for diversity
    s2_all = pl.read_csv(BASE + "train_source2.tsv", separator="\t")
    s3_all = pl.read_csv(BASE + "train_source3.tsv", separator="\t")
    s2_dist = s2_all.sample(n=min(N_DISTRACTORS, len(s2_all)), seed=42)
    s3_dist = s3_all.sample(n=min(N_DISTRACTORS, len(s3_all)), seed=42)

    pool_df = pl.concat([s2_targets, s3_targets, s2_dist, s3_dist]).unique(subset=["entity_id"])
    print(f"  Total candidate pool size: {len(pool_df):,} records")

    pool_dict: Dict[str, dict] = {r["entity_id"]: r for r in pool_df.to_dicts()}
    pool_ids: List[str] = pool_df["entity_id"].to_list()  # index -> entity_id mapping

    print("  Building blocking index ...")
    t0 = time.time()
    idx = build_index_from_df(pool_df)
    print(f"  Index built in {time.time() - t0:.2f}s")

    # ── 4. Extract Pairwise Features ─────────────────────────────────────────
    print(f"\n[4/5] Extracting pairwise features for {len(s1_ids):,} entities …")
    t0 = time.time()

    rows = []
    for i, sid in enumerate(s1_ids):
        s1_rec = s1_dict.get(sid)
        if not s1_rec:
            continue

        s1_ctry = (s1_rec.get("country") or "").strip()
        s1_core, _ = clean_name(s1_rec.get("business_name"))
        _, s1_dist, s1_nums = clean_address(s1_rec.get("business_address"))

        # Retrieve candidates — query_indices returns (int_idx, score) pairs
        raw_cands = idx.query_indices(s1_ctry, s1_core, s1_dist, s1_nums, top_k=TOP_K)
        cands_scored = [(pool_ids[idx_], bs) for idx_, bs in raw_cands]
        true_set = gt.get(sid, set())
        max_bs = cands_scored[0][1] if cands_scored else 1.0

        for cand_id, bs in cands_scored:
            cand_rec = pool_dict.get(cand_id)
            if not cand_rec:
                continue

            label = 1 if cand_id in true_set else 0
            feats = extract_features(s1_rec, cand_rec, blocking_score=bs, max_blocking_score=max_bs)

            row = {"s1_id": sid, "cand_id": cand_id, "label": label}
            for fn, fv in zip(FEATURE_NAMES, feats):
                row[fn] = fv
            rows.append(row)

        # Include any true target matches not caught by top-k
        retrieved_ids = {cid for cid, _ in cands_scored}
        missed_true = true_set - retrieved_ids
        for cand_id in missed_true:
            cand_rec = pool_dict.get(cand_id)
            if not cand_rec:
                continue
            feats = extract_features(s1_rec, cand_rec, blocking_score=0.0, max_blocking_score=1.0)
            row = {"s1_id": sid, "cand_id": cand_id, "label": 1}
            for fn, fv in zip(FEATURE_NAMES, feats):
                row[fn] = fv
            rows.append(row)

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    Progress: {i+1:,}/{len(s1_ids):,} entities processed | {len(rows):,} pairs ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Feature extraction complete: {len(rows):,} pairs generated in {elapsed:.1f}s ({len(rows)/elapsed:,.0f} pairs/sec)")

    # ── 5. Save Parquet ──────────────────────────────────────────────────────
    print(f"\n[5/5] Saving to {OUTPUT_PATH} …")
    out_df = pl.DataFrame(rows)
    out_df.write_parquet(OUTPUT_PATH)
    print(f"  Saved {len(out_df):,} rows × {len(out_df.columns)} columns to Parquet.")

    # ── Summary Stats ────────────────────────────────────────────────────────
    n_pos = out_df.filter(pl.col("label") == 1).height
    n_neg = out_df.filter(pl.col("label") == 0).height
    print()
    print("=" * 60)
    print("  PAIRWISE DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total pairs:      {len(out_df):>10,}")
    print(f"  Positive matches: {n_pos:>10,} ({n_pos/len(out_df)*100:.1f}%)")
    print(f"  Hard negatives:   {n_neg:>10,} ({n_neg/len(out_df)*100:.1f}%)")
    print(f"  Imbalance ratio:  1 : {n_neg/n_pos:.2f}")
    print("=" * 60)
    print("  ✓ PHASE 3 COMPLETE: Feature matrix successfully generated.")
    print("=" * 60)


if __name__ == "__main__":
    build_dataset()
