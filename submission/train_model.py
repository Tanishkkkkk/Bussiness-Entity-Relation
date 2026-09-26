"""
Phase 4 — Model Training: LightGBM Classifier with F0.5 Optimised Threshold.

Pipeline:
  1. Load train_pairs.parquet from Phase 3
  2. Split: 80% train / 20% val (stratified by label)
  3. Train LightGBM binary classifier with class_weight='balanced'
  4. Search for optimal dual threshold (t_match, t_nomatch) maximising macro-F0.5
  5. Save: model.pkl  +  thresholds.json  +  feature_importance.csv

Macro-F0.5 is Precision-weighted (Beta=0.5 → Precision counts 2x more than Recall).

NOTE: This script has NO torch/GPU dependency — LightGBM runs on CPU by default.
"""

import os
import sys
import json
import time
import pickle

import polars as pl
import numpy as np

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
sys.stdout.reconfigure(encoding="utf-8")

try:
    import lightgbm as lgb
    LGBM = True
except ImportError:
    LGBM = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    SKLEARN = True
except ImportError:
    SKLEARN = False

from features import FEATURE_NAMES

OUTPUT_DIR = "dataset/processed"
MODEL_PATH  = f"{OUTPUT_DIR}/model.pkl"
THRESH_PATH = f"{OUTPUT_DIR}/thresholds.json"
FEAT_PATH   = f"{OUTPUT_DIR}/feature_importance.csv"

SEED = 42
TEST_FRAC = 0.2


# ─── F0.5 Metrics ─────────────────────────────────────────────────────────────

def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    if precision + recall == 0:
        return 0.0
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def macro_f05(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-F0.5 across two classes (match / no-match).
    Both classes treated equally in macro average.
    """
    scores = []
    for cls in [0, 1]:
        tp = ((y_pred == cls) & (y_true == cls)).sum()
        fp = ((y_pred == cls) & (y_true != cls)).sum()
        fn = ((y_pred != cls) & (y_true == cls)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(fbeta(prec, rec))
    return float(np.mean(scores))


def optimise_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Grid search single threshold in [0.1, 0.9] → maximise macro-F0.5."""
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (y_prob >= t).astype(int)
        sc = macro_f05(y_true, pred)
        if sc > best_score:
            best_score, best_t = sc, float(t)
    return best_t, best_score


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    path = f"{OUTPUT_DIR}/train_pairs.parquet"
    print(f"Loading {path} …")
    df = pl.read_parquet(path)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    X = df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y = df["label"].to_numpy().astype(np.int32)

    # Stratified split by s1_id (keep all pairs for an entity in same split)
    s1_ids = df["s1_id"].to_numpy()
    unique_ids = np.unique(s1_ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_ids)
    split = int(len(unique_ids) * (1 - TEST_FRAC))
    train_ids = set(unique_ids[:split])
    val_ids   = set(unique_ids[split:])

    train_mask = np.array([sid in train_ids for sid in s1_ids])
    val_mask   = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f"  Train: {X_train.shape[0]:,}  (pos={y_train.sum():,}  neg={(y_train==0).sum():,})")
    print(f"  Val:   {X_val.shape[0]:,}  (pos={y_val.sum():,}   neg={(y_val==0).sum():,})")
    return X_train, y_train, X_val, y_val


# ─── LightGBM Trainer ─────────────────────────────────────────────────────────

def train_lgbm(X_train, y_train, X_val, y_val):
    print("\nTraining LightGBM …")
    pos_w = (y_train == 0).sum() / (y_train == 1).sum()  # balance
    params = {
        "objective":       "binary",
        "metric":          "binary_logloss",
        "learning_rate":   0.05,
        "num_leaves":      63,
        "min_child_samples": 20,
        "n_estimators":    500,
        "scale_pos_weight": pos_w,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "random_state":    SEED,
        "n_jobs":          -1,
        "verbose":         -1,
    }
    model = lgb.LGBMClassifier(**params)
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    print(f"  Trained in {time.time()-t0:.1f}s  (best iter: {model.best_iteration_})")
    return model


# ─── Fallback: sklearn GBM ─────────────────────────────────────────────────────

def train_sklearn(X_train, y_train):
    print("\nTraining sklearn GradientBoostingClassifier (LightGBM not available) …")
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=SEED,
        verbose=1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHASE 4: MODEL TRAINING")
    print("=" * 60)

    X_train, y_train, X_val, y_val = load_data()

    if LGBM:
        model = train_lgbm(X_train, y_train, X_val, y_val)
    elif SKLEARN:
        model = train_sklearn(X_train, y_train)
        X_val_final, y_val_final = X_val, y_val
    else:
        raise RuntimeError("Neither lightgbm nor sklearn is available. Run: pip install lightgbm")

    # ── Calibrate threshold on validation set ─────────────────────────────────
    print("\nCalibrating threshold on validation set …")
    y_prob = model.predict_proba(X_val)[:, 1]
    best_t, best_f05 = optimise_threshold(y_val, y_prob)
    print(f"  Best threshold: {best_t:.2f}  →  macro-F0.5 = {best_f05:.4f}")

    # Also report at default 0.5
    default_pred = (y_prob >= 0.5).astype(int)
    default_f05  = macro_f05(y_val, default_pred)
    print(f"  Default (t=0.50) macro-F0.5 = {default_f05:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    print(f"\nSaving model → {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    thresh = {"match_threshold": best_t, "val_macro_f05": best_f05}
    with open(THRESH_PATH, "w") as f:
        json.dump(thresh, f, indent=2)
    print(f"Saved thresholds → {THRESH_PATH}")

    # ── Feature importance ────────────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        imp_df = pl.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": imp.tolist(),
        }).sort("importance", descending=True)
        imp_df.write_csv(FEAT_PATH)
        print(f"\nTop-10 Features:")
        for row in imp_df.head(10).iter_rows(named=True):
            bar = "█" * int(row["importance"] / imp.max() * 30)
            print(f"  {row['feature']:<22} {bar} {row['importance']:.1f}")

    print()
    print("=" * 60)
    print(f"  ✓ PHASE 4 COMPLETE")
    print(f"    Val macro-F0.5:   {best_f05:.4f}")
    print(f"    Match threshold:  {best_t:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
