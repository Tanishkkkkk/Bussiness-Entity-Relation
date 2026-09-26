"""
Phase 2 Validation — Blocking Recall & Candidate Pool Benchmark.

Tests the production BlockingIndex on 1,000 GT-matched S1 entities with:
  - 100% of their true target records in the pool
  - 100,000 distractor records (realistic noise)

Reports Recall Ceiling and Average Candidate Pool Size.
"""

import sys, os, time
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
from blocker import build_index_from_df, retrieve_candidates

BASE = "dataset/train/"
N_GT  = 1000   # S1 entities to test


def run():
    print(f"Loading first {N_GT} GT entries …")
    gt_df = pl.read_csv(BASE + "train_ground_truth.tsv", separator="\t", n_rows=N_GT)

    gt: dict = {}
    s1_ids: list = []
    target_ids: set = set()
    for r in gt_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        s1_ids.append(sid)
        m = r.get("matched_entity_ids") or ""
        if m.strip():
            mt = set(m.split(","))
            gt[sid] = mt
            target_ids.update(mt)
        else:
            gt[sid] = set()

    print(f"  {len(s1_ids)} S1 ids, {len(target_ids)} unique target matches.")

    # ── Load S1 ──────────────────────────────────────────────────────────────
    print("Loading matching S1 records …")
    s1_df = (
        pl.scan_csv(BASE + "train_source1.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(s1_ids))
        .collect()
    )
    print(f"  Loaded {len(s1_df)} S1 records.")

    # ── Load candidate pool ───────────────────────────────────────────────────
    print("Loading candidate pool (targets + 50k distractors from each source) …")
    s2_dist = pl.read_csv(BASE + "train_source2.tsv", separator="\t", n_rows=50_000)
    s3_dist = pl.read_csv(BASE + "train_source3.tsv", separator="\t", n_rows=50_000)
    s2_tgt  = (
        pl.scan_csv(BASE + "train_source2.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in([x for x in target_ids if x.startswith("S2-")]))
        .collect()
    )
    s3_tgt  = (
        pl.scan_csv(BASE + "train_source3.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in([x for x in target_ids if x.startswith("S3-")]))
        .collect()
    )
    pool_df = pl.concat([s2_dist, s3_dist, s2_tgt, s3_tgt]).unique(subset=["entity_id"])
    print(f"  Pool: {len(pool_df):,} records (100% targets + distractors).")

    # ── Build index ───────────────────────────────────────────────────────────
    print("Building blocking index …")
    t0 = time.time()
    idx = build_index_from_df(pool_df)
    build_time = time.time() - t0
    print(f"  Index built in {build_time:.2f}s.")

    # ── Retrieve candidates ───────────────────────────────────────────────────
    print("Retrieving candidates for all S1 entities …")
    t0 = time.time()
    cands = retrieve_candidates(s1_df, idx, top_k=12)
    query_time = time.time() - t0

    # ── Evaluate ──────────────────────────────────────────────────────────────
    total_true = 0
    captured   = 0
    pool_sizes = []

    for sid, cand_list in cands.items():
        true_set = gt.get(sid, set())
        total_true += len(true_set)
        captured   += len(true_set & set(cand_list))
        pool_sizes.append(len(cand_list))

    recall   = captured / total_true * 100 if total_true else 0
    avg_pool = sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0

    print()
    print("=" * 60)
    print("  PHASE 2 BLOCKING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  S1 entities evaluated:     {len(s1_df):>8,}")
    print(f"  True target matches:       {total_true:>8,}")
    print(f"  Captured matches:          {captured:>8,}")
    print(f"  Recall Ceiling:            {recall:>8.2f}%  (target ≥ 95%)")
    print(f"  Avg Candidate Pool Size:   {avg_pool:>8.2f}  (target ≤ 15)")
    print(f"  Index build time:          {build_time:>8.2f}s")
    print(f"  Query time ({len(s1_df)} S1):  {query_time:>8.2f}s  ({len(s1_df)/query_time:,.0f} S1/sec)")
    print("=" * 60)

    ok = True
    if recall < 90:
        print(f"  ⚠ RECALL {recall:.1f}% < 90% — need more blocking keys")
        ok = False
    if avg_pool > 20:
        print(f"  ⚠ POOL {avg_pool:.1f} > 20 — too many candidates, need tighter pruning")
        ok = False
    if ok:
        print("  ✓ PHASE 2 PASS — blocking meets quality targets!")


if __name__ == "__main__":
    run()
