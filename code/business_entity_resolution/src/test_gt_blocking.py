"""
Test multi-key blocking accuracy on 1000 S1 records with their full GT matches.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
import polars as pl
from normalizer import clean_name, clean_address

def test_full_gt_blocking(n_s1: int = 1000):
    print(f"Loading first {n_s1} GT entries and finding their matching S1, S2, and S3 records...")
    base = "dataset/train/"
    
    # 1. Load GT first
    gt_df = pl.read_csv(base + "train_ground_truth.tsv", separator="\t", n_rows=n_s1)
    gt = {}
    target_ids = set()
    s1_ids = []
    
    for r in gt_df.iter_rows(named=True):
        s1_id = r["source1_entity_id"]
        s1_ids.append(s1_id)
        m = r["matched_entity_ids"]
        if m and m.strip():
            matches = set(m.split(","))
            gt[s1_id] = matches
            target_ids.update(matches)
        else:
            gt[s1_id] = set()
            
    print(f"Loaded {len(s1_ids)} S1 IDs. Total True Matches to find: {len(target_ids)} target entities")
    
    # 2. Load the exact S1 records from train_source1 using scan_csv filter
    print("Loading corresponding S1 records from train_source1.tsv...")
    s1_df = pl.scan_csv(base + "train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(s1_ids)).collect()
    print(f"Loaded {len(s1_df)} S1 records.")
    
    # 3. Find and load all true target records from S2 and S3, PLUS 50,000 random distractor records
    print("Loading target records and distractors from S2 and S3...")
    s2_targets = set(x for x in target_ids if x.startswith("S2-"))
    s3_targets = set(x for x in target_ids if x.startswith("S3-"))
    
    s2_all = pl.read_csv(base + "train_source2.tsv", separator="\t", n_rows=50000)
    s3_all = pl.read_csv(base + "train_source3.tsv", separator="\t", n_rows=50000)
    
    s2_matched = pl.scan_csv(base + "train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s2_targets))).collect()
    s3_matched = pl.scan_csv(base + "train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s3_targets))).collect()
    
    pool_df = pl.concat([s2_all, s3_all, s2_matched, s3_matched]).unique(subset=["entity_id"])
    print(f"Total candidate pool: {len(pool_df):,} records (contains 100% of targets + 100k distractors)")
    
    # 4. Build Inverted Index
    print("Building Inverted Index...")
    inverted_index = defaultdict(list)
    freq = defaultdict(int)
    
    temp_keys = []
    for r in pool_df.iter_rows(named=True):
        cid = r["entity_id"]
        c = r["country"] or ""
        core, _ = clean_name(r["business_name"])
        clean_a, tokens_a, nums_a = clean_address(r["business_address"])
        
        words = core.split()
        keys = []
        if words:
            w1 = words[0]
            if len(w1) >= 3:
                keys.append((c, f"w1_{w1[:4]}"))
            if len(words) > 1 and len(words[1]) >= 3:
                keys.append((c, f"w2_{words[1][:4]}"))
        # 5-6 digit pin/zip
        for n in nums_a:
            if 5 <= len(n) <= 6:
                keys.append((c, f"zip_{n}"))
            elif len(n) >= 2:
                for t in tokens_a:
                    if len(t) >= 4 and not t.isdigit():
                        keys.append((c, f"num_{n}_{t[:4]}"))
                        break
        temp_keys.append((cid, keys))
        for k in keys:
            freq[k] += 1
            
    MAX_BUCKET = 500
    for cid, keys in temp_keys:
        for k in keys:
            if freq[k] <= MAX_BUCKET:
                inverted_index[k].append(cid)
                
    print(f"Index built with {len(inverted_index):,} active keys.")
    
    # 5. Query S1 entities
    print("Evaluating Blocking on S1 records...")
    max_cands = 12
    captured = 0
    total_targets = 0
    cands_per_s1 = []
    
    for r in s1_df.iter_rows(named=True):
        s1_id = r["entity_id"]
        c = r["country"] or ""
        core, _ = clean_name(r["business_name"])
        clean_a, tokens_a, nums_a = clean_address(r["business_address"])
        
        true_set = gt.get(s1_id, set())
        total_targets += len(true_set)
        
        scores = defaultdict(float)
        words = core.split()
        if words:
            w1 = words[0]
            if len(w1) >= 3:
                for cid in inverted_index.get((c, f"w1_{w1[:4]}"), []):
                    scores[cid] += 3.0
            if len(words) > 1 and len(words[1]) >= 3:
                for cid in inverted_index.get((c, f"w2_{words[1][:4]}"), []):
                    scores[cid] += 2.0
                    
        for n in nums_a:
            if 5 <= len(n) <= 6:
                for cid in inverted_index.get((c, f"zip_{n}"), []):
                    scores[cid] += 2.5
            elif len(n) >= 2:
                for t in tokens_a:
                    if len(t) >= 4 and not t.isdigit():
                        for cid in inverted_index.get((c, f"num_{n}_{t[:4]}"), []):
                            scores[cid] += 2.0
                        break
                        
        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_cands]
        retrieved = set(x[0] for x in top_k)
        cands_per_s1.append(len(retrieved))
        
        hits = true_set.intersection(retrieved)
        captured += len(hits)
        
    recall = captured / total_targets * 100 if total_targets else 0
    avg_pool = sum(cands_per_s1) / len(cands_per_s1)
    
    print("\n" + "="*60)
    print("BLOCKING RESULTS ON GROUND TRUTH")
    print("="*60)
    print(f"S1 Records Tested:         {len(s1_df):,}")
    print(f"Total True Target Matches: {total_targets:,}")
    print(f"Captured Target Matches:   {captured:,}")
    print(f"Recall Ceiling:            {recall:.2f}%")
    print(f"Average Candidate Pool:    {avg_pool:.2f} per S1 (Target <= 15)")
    print("="*60)

if __name__ == "__main__":
    test_full_gt_blocking(1000)
