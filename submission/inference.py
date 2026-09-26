"""
Phase 5 — High-Speed Vectorized Test Inference Engine.

Key optimizations:
  1. Country-partitioned candidate indexing (peak RAM < 700MB).
  2. Pre-computed normalized candidate representations (0 redundant normalization inside loop).
  3. Chunk-batched LightGBM matrix scoring (10,000 queries scored in single C-API call).
  4. Generates:
     - output/candidate_pairs.tsv
     - output/matching_results.tsv
"""

import os
import sys
import json
import time
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import polars as pl
import numpy as np

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
sys.stdout.reconfigure(encoding="utf-8")

from blocker import BlockingIndex, build_index_from_lists
from features import extract_features_precomputed, FEATURE_NAMES
from normalizer import clean_name, clean_address

MODEL_PATH   = "dataset/processed/model.pkl"
THRESH_PATH  = "dataset/processed/thresholds.json"
OUTPUT_DIR   = "output"
CAND_OUT     = f"{OUTPUT_DIR}/candidate_pairs.tsv"
MATCH_OUT    = f"{OUTPUT_DIR}/matching_results.tsv"

TOP_K        = 12
CHUNK_EVAL   = 10_000   # batch size for vectorized LightGBM scoring


def run_inference(test_dir: str = "dataset/test"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60, flush=True)
    print("  PHASE 5: HIGH-SPEED TEST INFERENCE & SUBMISSION PIPELINE", flush=True)
    print("=" * 60, flush=True)

    # ── 1. Load Model & Threshold ─────────────────────────────────────────────
    print(f"\n[1/4] Loading trained LightGBM model from {MODEL_PATH} …", flush=True)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(THRESH_PATH, "r") as f:
        thresh_info = json.load(f)
    threshold = float(thresh_info.get("match_threshold", 0.84))
    print(f"  Model loaded. Classification threshold = {threshold:.2f}", flush=True)

    # ── 2. Read Source 1 Metadata & Order ─────────────────────────────────────
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"\n[2/4] Reading Source 1 from {s1_path} …", flush=True)
    t0 = time.time()
    s1_df = pl.read_csv(s1_path, separator="\t")
    n_s1 = len(s1_df)
    print(f"  Loaded {n_s1:,} Source 1 records in {time.time()-t0:.2f}s", flush=True)

    # Pre-normalize S1 entities
    print("  Pre-normalizing Source 1 entities …", flush=True)
    t_norm = time.time()
    s1_eids = s1_df["entity_id"].to_list()
    s1_names = s1_df["business_name"].to_list()
    s1_addrs = s1_df["business_address"].to_list()
    s1_ctrys = [(c or "Unknown").strip() for c in s1_df["country"].to_list()]
    del s1_df

    s1_norm = []
    s1_by_country = defaultdict(list)
    for i in range(n_s1):
        core_n, full_n = clean_name(s1_names[i])
        addr_s, dist_a, nums_s = clean_address(s1_addrs[i])
        ctry = s1_ctrys[i]
        norm_tuple = (s1_eids[i], core_n, full_n, addr_s, dist_a, nums_s, ctry)
        s1_norm.append(norm_tuple)
        s1_by_country[ctry].append(i)
    
    del s1_names, s1_addrs, s1_ctrys
    print(f"  Source 1 pre-normalized in {time.time()-t_norm:.2f}s across {len(s1_by_country)} countries.", flush=True)

    # Output lines aligned to original order
    cand_results: List[str] = [""] * n_s1
    match_results: List[str] = [""] * n_s1

    # ── 3. Load Candidates (Source 2 + Source 3) ─────────────────────────────
    print(f"\n[3/4] Loading candidates …", flush=True)
    t_pool = time.time()
    s2_df = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t")
    pool_df = pl.concat([s2_df, s3_df]).unique(subset=["entity_id"])
    del s2_df, s3_df
    print(f"  Total unique candidates: {len(pool_df):,} records ({time.time()-t_pool:.2f}s)", flush=True)

    # ── 4. Process Country Partitions ─────────────────────────────────────────
    print(f"\n[4/4] Running inference by country partition …", flush=True)
    total_candidates_count = 0
    total_matches_count = 0
    t_inf_start = time.time()

    for ctry_name, orig_indices in s1_by_country.items():
        t_c_start = time.time()
        print(f"\n  --- Country: {ctry_name} ({len(orig_indices):,} S1 entities) ---", flush=True)

        c_pool = pool_df.filter(pl.col("country") == ctry_name)
        n_cands = len(c_pool)
        print(f"    Candidate pool: {n_cands:,} records", flush=True)

        if n_cands == 0:
            for o_idx in orig_indices:
                sid = s1_norm[o_idx][0]
                cand_results[o_idx] = f"{sid}\t\n"
                match_results[o_idx] = f"{sid}\t\n"
            continue

        c_eids_raw  = c_pool["entity_id"].to_list()
        c_names_raw = c_pool["business_name"].to_list()
        c_addrs_raw = c_pool["business_address"].to_list()
        c_ctrys_raw = c_pool["country"].to_list()
        del c_pool

        # Pre-normalize candidates for this country
        print(f"    Pre-normalizing {n_cands:,} candidate records …", flush=True)
        t_cn = time.time()
        c_cores = []
        c_fulls = []
        c_addrs = []
        c_dists = []
        c_nums  = []
        for j in range(n_cands):
            core_n, full_n = clean_name(c_names_raw[j])
            addr_s, dist_a, nums_s = clean_address(c_addrs_raw[j])
            c_cores.append(core_n)
            c_fulls.append(full_n)
            c_addrs.append(addr_s)
            c_dists.append(dist_a)
            c_nums.append(nums_s)
        print(f"    Candidate pre-normalization done in {time.time()-t_cn:.2f}s", flush=True)

        # Build index
        t_idx = time.time()
        idx = BlockingIndex(max_bucket=500)
        for j in range(n_cands):
            idx.add_record(j, ctry_name, c_cores[j], c_dists[j], c_nums[j])
        idx.finalize()
        print(f"    Index constructed in {time.time()-t_idx:.2f}s", flush=True)

        # Query and evaluate in batches
        c_matches = 0
        c_cands = 0
        n_queries = len(orig_indices)

        for chunk_start in range(0, n_queries, CHUNK_EVAL):
            chunk_indices = orig_indices[chunk_start : chunk_start + CHUNK_EVAL]
            
            # 1. Blocking retrieval & pair feature extraction
            batch_pairs_feats = []
            batch_pair_meta = []   # (s1_orig_idx, cand_eid)

            for o_idx in chunk_indices:
                sid, s1_core, s1_full, s1_addr, s1_dist, s1_nums, s1_c = s1_norm[o_idx]
                
                cands_scored = idx.query_indices(ctry_name, s1_core, s1_dist, s1_nums, top_k=TOP_K)
                cand_id_list = [c_eids_raw[c[0]] for c in cands_scored]
                c_cands += len(cand_id_list)

                cand_results[o_idx] = f"{sid}\t{','.join(cand_id_list)}\n"

                if cands_scored:
                    max_bs = cands_scored[0][1]
                    for c_idx, bs in cands_scored:
                        feats = extract_features_precomputed(
                            s1_core, s1_full, s1_addr, s1_dist, s1_nums, s1_c,
                            c_cores[c_idx], c_fulls[c_idx], c_addrs[c_idx], c_dists[c_idx], c_nums[c_idx], ctry_name,
                            blocking_score=bs, max_blocking_score=max_bs
                        )
                        batch_pairs_feats.append(feats)
                        batch_pair_meta.append((o_idx, c_eids_raw[c_idx]))

            # 2. Vectorized LightGBM scoring
            s1_matches_map = defaultdict(list)
            if batch_pairs_feats:
                X_batch = np.array(batch_pairs_feats, dtype=np.float32)
                probs = model.predict_proba(X_batch)[:, 1]
                for (o_idx, cid), prob in zip(batch_pair_meta, probs):
                    if prob >= threshold:
                        s1_matches_map[o_idx].append(cid)

            # 3. Assemble matching lines
            for o_idx in chunk_indices:
                sid = s1_norm[o_idx][0]
                matched_list = s1_matches_map.get(o_idx, [])
                c_matches += len(matched_list)
                match_results[o_idx] = f"{sid}\t{','.join(matched_list)}\n"

            progress = min(chunk_start + CHUNK_EVAL, n_queries)
            elapsed_c = time.time() - t_c_start
            speed = progress / elapsed_c if elapsed_c > 0 else 0
            print(f"    Progress: {progress:>8,}/{n_queries:,} | Matches: {c_matches:>7,} | Speed: {speed:>6.0f} S1/sec", flush=True)

        total_candidates_count += c_cands
        total_matches_count += c_matches
        del c_eids_raw, c_names_raw, c_addrs_raw, c_ctrys_raw, c_cores, c_fulls, c_addrs, c_dists, c_nums, idx
        print(f"    ✓ Done {ctry_name}: {c_matches:,} matches from {c_cands:,} candidates in {time.time()-t_c_start:.2f}s", flush=True)

    del pool_df, s1_norm

    # ── 5. Write Complete Outputs ─────────────────────────────────────────────
    print(f"\n[5/5] Writing final submission files …", flush=True)
    t_w = time.time()

    with open(CAND_OUT, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.writelines(cand_results)

    with open(MATCH_OUT, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.writelines(match_results)

    total_time = time.time() - t_inf_start
    print(f"  Files written in {time.time()-t_w:.2f}s:", flush=True)
    print(f"    • {CAND_OUT}  ({len(cand_results):,} rows, avg {total_candidates_count/n_s1:.2f} candidates/entity)", flush=True)
    print(f"    • {MATCH_OUT} ({len(match_results):,} rows, total {total_matches_count:,} matches)", flush=True)
    print(f"  Overall inference speed: {n_s1/total_time:,.0f} S1/sec across all {n_s1:,} test entities!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_inference()
