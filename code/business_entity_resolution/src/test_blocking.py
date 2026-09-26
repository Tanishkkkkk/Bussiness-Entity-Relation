"""
Experimental benchmark to test Blocking & Candidate Generation on Ground Truth.
Evaluates Candidate Set Size, Reduction Ratio, and Recall Ceiling.
"""

import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

sys.stdout.reconfigure(encoding="utf-8")
import polars as pl
from normalizer import clean_name, clean_address, extract_blocking_tokens


def run_blocking_benchmark(sample_size: int = 10000):
    print(f"Loading {sample_size} training records for blocking benchmark...")
    base_dir = "dataset/train/"

    # Load S1 sample
    s1_df = pl.read_csv(
        base_dir + "train_source1.tsv", separator="\t", n_rows=sample_size
    )
    s1_ids = set(s1_df["entity_id"].to_list())

    # Load GT
    gt_df = pl.read_csv(
        base_dir + "train_ground_truth.tsv", separator="\t", n_rows=sample_size
    )
    gt: Dict[str, Set[str]] = {}
    total_true_matches = 0
    for row in gt_df.iter_rows(named=True):
        m = row["matched_entity_ids"]
        if m and m.strip():
            matches = set(m.split(","))
            gt[row["source1_entity_id"]] = matches
            total_true_matches += len(matches)
        else:
            gt[row["source1_entity_id"]] = set()

    print(f"Total S1 entities: {len(s1_ids)}")
    print(f"Total True Matches to find: {total_true_matches}")

    # Load S2 and S3 records (load enough to contain all candidates for this sample)
    print("Loading S2 and S3 records for evaluation...")
    s2_df = pl.read_csv(base_dir + "train_source2.tsv", separator="\t", n_rows=200000)
    s3_df = pl.read_csv(base_dir + "train_source3.tsv", separator="\t", n_rows=200000)

    # Let's also ensure all GT matches are included in the pool so recall is faithfully tested
    all_gt_target_ids = set()
    for targets in gt.values():
        all_gt_target_ids.update(targets)

    # Preprocess S2 and S3 into Candidate Pool
    print(f"Building candidate pool from S2 ({len(s2_df):,}) and S3 ({len(s3_df):,})...")
    candidates = {}
    # Token to Candidate IDs inverted index
    inverted_index = defaultdict(list)

    for df in [s2_df, s3_df]:
        for row in df.iter_rows(named=True):
            cid = row["entity_id"]
            country = row["country"] or ""
            core_name, _ = clean_name(row["business_name"])
            clean_addr, tokens_addr, num_tokens = clean_address(row["business_address"])

            candidates[cid] = {
                "name": core_name,
                "addr": clean_addr,
                "nums": num_tokens,
                "country": country,
            }

            # Indexing keys:
            # 1. First 4 chars of core name
            name_words = core_name.split()
            if name_words:
                w1 = name_words[0]
                if len(w1) >= 3:
                    inverted_index[(country, f"w1_{w1[:4]}")].append(cid)
                if len(name_words) > 1 and len(name_words[1]) >= 3:
                    inverted_index[(country, f"w2_{name_words[1][:4]}")].append(cid)

            # 2. Numeric tokens (house numbers, postal codes)
            for n in num_tokens:
                if len(n) >= 4:  # typical zip / pin / unit code
                    inverted_index[(country, f"num_{n}")].append(cid)

    print(f"Inverted index built with {len(inverted_index):,} keys.")

    # Now retrieve candidates for each S1 entity
    print("Querying candidate generator...")
    start_time = time.time()
    candidates_per_s1 = {}
    captured_matches = 0
    total_eval_matches = 0

    max_candidates_per_s1 = 12

    for row in s1_df.iter_rows(named=True):
        s1_id = row["entity_id"]
        country = row["country"] or ""
        core_name, _ = clean_name(row["business_name"])
        clean_addr, tokens_addr, num_tokens = clean_address(row["business_address"])

        true_targets = gt.get(s1_id, set())
        # Only count GT targets that were present in our candidate pool
        valid_targets = true_targets.intersection(set(candidates.keys()))
        total_eval_matches += len(valid_targets)

        # Retrieve candidates from inverted index
        cand_scores = defaultdict(float)
        name_words = core_name.split()

        if name_words:
            w1 = name_words[0]
            if len(w1) >= 3:
                for cid in inverted_index.get((country, f"w1_{w1[:4]}"), []):
                    cand_scores[cid] += 2.0
            if len(name_words) > 1 and len(name_words[1]) >= 3:
                for cid in inverted_index.get((country, f"w2_{name_words[1][:4]}"), []):
                    cand_scores[cid] += 1.5

        for n in num_tokens:
            if len(n) >= 4:
                for cid in inverted_index.get((country, f"num_{n}"), []):
                    cand_scores[cid] += 1.0

        # Sort and take top K
        sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[
            :max_candidates_per_s1
        ]
        retrieved_ids = [c[0] for c in sorted_cands]
        candidates_per_s1[s1_id] = retrieved_ids

        # Check hits
        hits = valid_targets.intersection(set(retrieved_ids))
        captured_matches += len(hits)

    elapsed = time.time() - start_time
    recall = (captured_matches / total_eval_matches * 100) if total_eval_matches > 0 else 0
    avg_cands = sum(len(c) for c in candidates_per_s1.values()) / len(candidates_per_s1)

    print("\n" + "=" * 60)
    print("BLOCKING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total S1 entities evaluated:    {len(s1_df):,}")
    print(f"Target matches in pool:         {total_eval_matches:,}")
    print(f"Captured matches:               {captured_matches:,}")
    print(f"Recall Ceiling:                 {recall:.2f}%")
    print(f"Average Candidate Pool Size:    {avg_cands:.2f} per S1 (target <= 15)")
    print(f"Query Time:                     {elapsed:.2f}s ({len(s1_df)/elapsed:,.0f} S1/sec)")
    print("=" * 60)


if __name__ == "__main__":
    run_blocking_benchmark(5000)
