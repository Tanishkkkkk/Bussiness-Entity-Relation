"""
Entity-level Macro-F0.5 Evaluator.

The competition evaluates at the ENTITY level:
  For each S1 entity i:
    - GT_i   = set of true matching S2/S3 IDs
    - PRED_i = set of predicted matching S2/S3 IDs (those with score >= threshold)
    - F0.5_i = (1.25 * P_i * R_i) / (0.25 * P_i + R_i)  [0 if both zero]
  Final score = mean(F0.5_i) across ALL S1 entities

Our previous metric was pair-level binary F0.5 - completely different!
This script finds the true optimal threshold for entity-level evaluation.
"""

import sys, os, json, pickle
import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
sys.stdout.reconfigure(encoding="utf-8")

from features import FEATURE_NAMES

MODEL_PATH  = "dataset/processed/model.pkl"
THRESH_PATH = "dataset/processed/thresholds.json"
PAIRS_PATH  = "dataset/processed/train_pairs.parquet"
GT_PATH     = "dataset/train/train_ground_truth.tsv"
SEED = 42


def f05_entity(pred: set, gt: set) -> float:
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    tp   = len(pred & gt)
    prec = tp / len(pred)
    rec  = tp / len(gt)
    if prec + rec == 0:
        return 0.0
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def macro_f05_entity(val_s1ids, val_cand_ids, val_scores, gt_dict, threshold):
    pred_dict = {}
    for s1, cand, sc in zip(val_s1ids, val_cand_ids, val_scores):
        if sc >= threshold:
            pred_dict.setdefault(s1, set()).add(cand)
    all_s1 = set(val_s1ids)
    scores = [f05_entity(pred_dict.get(s1, set()), gt_dict.get(s1, set())) for s1 in all_s1]
    return float(np.mean(scores))


def main():
    print("=" * 60)
    print("  ENTITY-LEVEL MACRO-F0.5 THRESHOLD OPTIMIZER")
    print("=" * 60)

    print("\n[1] Loading model ...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("[2] Loading train_pairs.parquet ...")
    df = pl.read_parquet(PAIRS_PATH)
    print(f"  {len(df):,} pairs  (pos={df['label'].sum():,}  neg={(df['label']==0).sum():,})")

    print("[3] Reproducing val split (80/20 by s1_id) ...")
    s1_ids = df["s1_id"].to_numpy()
    unique_ids = np.unique(s1_ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_ids)
    split = int(len(unique_ids) * 0.8)
    val_ids = set(unique_ids[split:])
    val_mask = np.array([sid in val_ids for sid in s1_ids])
    df_val = df.filter(pl.Series(val_mask))
    print(f"  Val set: {len(df_val):,} pairs  ({df_val['label'].sum():,} positives)")

    print("[4] Scoring val pairs ...")
    X_val        = df_val.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    val_scores   = model.predict_proba(X_val)[:, 1]
    val_s1ids    = df_val["s1_id"].to_numpy()
    val_cand_ids = df_val["cand_id"].to_numpy()

    print("[5] Loading ground truth ...")
    gt = pl.read_csv(GT_PATH, separator="\t")
    gt_dict = {}
    for row in gt.to_dicts():
        sid = row["source1_entity_id"]
        raw = row.get("matched_entity_ids") or ""
        gt_dict[sid] = set(raw.split(",")) if raw.strip() else set()
    print(f"  {len(gt_dict):,} GT entities")

    print("\n[6] Threshold sweep (entity-level Macro-F0.5) ...")
    print(f"  {'Threshold':>10}  {'Entity-F0.5':>13}  {'%Matched':>9}")
    print("  " + "-" * 38)

    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.40, 0.98, 0.02):
        sc = macro_f05_entity(val_s1ids, val_cand_ids, val_scores, gt_dict, t)
        n_matched = sum(1 for s1 in set(val_s1ids)
                        if any(sc_ >= t for sc_, s1_ in zip(val_scores, val_s1ids) if s1_ == s1))
        pct = 100 * sum(1 for sc_ in val_scores if sc_ >= t) / len(set(val_s1ids))
        marker = " << BEST" if sc > best_score else ""
        if sc > best_score:
            best_score, best_t = sc, float(t)
        print(f"  {t:10.2f}  {sc:13.4f}  {pct:8.1f}%{marker}")

    print(f"\nBest entity-level threshold: {best_t:.2f}  F0.5 = {best_score:.4f}")

    old_thresh_info = json.load(open(THRESH_PATH))
    old_t = old_thresh_info.get("match_threshold", 0.84)
    old_entity = macro_f05_entity(val_s1ids, val_cand_ids, val_scores, gt_dict, old_t)
    print(f"Old threshold {old_t:.2f} -> entity-level F0.5 = {old_entity:.4f}")
    print(f"New threshold {best_t:.2f} -> entity-level F0.5 = {best_score:.4f}")

    new_info = {"match_threshold": best_t, "val_entity_f05": best_score,
                "val_pair_f05": old_thresh_info.get("val_macro_f05", 0)}
    with open(THRESH_PATH, "w") as f:
        json.dump(new_info, f, indent=2)
    print(f"\nSaved updated threshold to {THRESH_PATH}")
    print("Re-run inference.py to generate new submission.")


if __name__ == "__main__":
    main()
