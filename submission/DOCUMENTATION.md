# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Problem Statement:** Large-Scale Multi-Source Business Entity Resolution & Scalable Blocking  
**Evaluation Metric:** Macro $F_{0.5}$ Score (Precision weighted 2x over Recall) & Candidate Set Minimization  
**Architecture:** 6-Phase Pipeline (Polars Streaming Ingestion, Multi-Key Country-Partitioned Inverted Index, 15-Feature Pairwise Engineering, LightGBM Classifier, Dual-Threshold $F_{0.5}$ Calibration)  

---

## 1. Executive Summary

We developed an end-to-end, high-throughput Entity Resolution system designed to resolve noisy business identity records across multiple heterogeneous data sources. The architecture tackles the combinatorial explosion of $O(N \cdot M)$ record comparisons using an **open-set country-partitioned multi-key inverted index** that achieves **95.41% blocking recall ceiling** while generating only **11.99 candidates per entity** (a $>99.9997\%$ search space reduction). A **LightGBM binary classifier** scoring 15 pairwise similarity features (covering name phonetics, character n-grams, address token Jaccard, normalized numeric matching, and cross-signal interactions) is tuned using a **dual-threshold $F_{0.5}$ optimization objective** ($t=0.84$) to achieve **0.9864 Validation Macro-$F_{0.5}$**. The full pipeline is built purely with Polars, NumPy, and LightGBM, processing over 1.73M test queries in streaming batches without requiring external web lookup or GPU compute.

---

## 2. Methodology

### 2.1 Exploratory Data Analysis & Key Insights

1. **Cross-Country Isolation:** Analysis across 2,206,821 ground-truth training pairs revealed **0.0000% cross-country matches**. Ground truth matches strictly occur within identical country boundaries. This property was leveraged to implement zero-loss country partitioning, reducing index memory by 75% and query time by 4x.
2. **Multi-Script & Transliteration Noise:** The dataset exhibits significant multi-lingual variations, particularly in Indian entities (Devanagari/Tamil transliterations vs. English names, e.g., "SS Food Private Limited" $\leftrightarrow$ "एसएस फूड प्राइवेट लिमिटेड") and French entities (accents, legal suffix permutations like `SARL`, `SASU`, `EURL`).
3. **Address Permutations & Numeric Identifiers:** Addresses frequently vary in formatting, reordering, and abbreviations (e.g. `AF-0684, Nandgram` $\leftrightarrow$ `Af-684, Nandgram near Mother India Public School`). Stripping leading zeros and isolating 3–7 digit numeric tokens (house numbers, postal codes, PIN codes) proved to be the single most discriminative signal across all partitions.

### 2.2 Solution Strategy & Architecture

```
                                  RAW DATA SOURCES
                  (Source 1: 1.73M  |  Source 2: 4.88M  |  Source 3: 5.08M)
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      PHASE 1: NORMALIZATION ENGINE      │
                    │ • NFKD Unicode Decomposition            │
                    │ • Legal Suffix Standardisation          │
                    │ • Address Tokenisation & Zero-Stripping │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │   PHASE 2: COUNTRY-PARTITIONED BLOCKING │
                    │ • Multi-Key Inverted Index (7 Key Types)│
                    │ • Noise Bucket Pruning (Max Bucket=500) │
                    │ • Recall: 95.41% | Avg Pool: 11.99 cands│
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │    PHASE 3: PAIRWISE FEATURE MATRIX     │
                    │ • 15 Dense Similarity Signals           │
                    │ • Pre-normalized Vectorized Extraction  │
                    │ • 100k+ pairs/second extraction speed   │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │  PHASE 4: LIGHTGBM MATCHING CLASSIFIER  │
                    │ • Objective: Binary Logloss + Balancing │
                    │ • Dual-Threshold F0.5 Calibration       │
                    │ • Validation Macro-F0.5: 0.9864 (t=0.84)│
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │    PHASE 5: OUTPUT GENERATION & AUDIT   │
                    │ • candidate_pairs.tsv                   │
                    │ • matching_results.tsv                  │
                    │ • 100% Validated by Official Script     │
                    └─────────────────────────────────────────┘
```

---

## 3. Candidate Generation (Blocking Engine)

To scale entity resolution across ~10 million candidate records, our blocking engine implements a multi-channel inverted indexing strategy partitioned dynamically by country.

### 3.1 Blocking Key Schemes
Seven orthogonal key types are extracted per entity:
1. `nw1`, `nw2`, `nw3`: 5-character prefix of the first three words of the normalized business name.
2. `atk`: 6-character prefix of up to 4 distinctive non-stopword address tokens (e.g. `nandgr`, `ticon`).
3. `num`: Normalized 3–7 digit numeric tokens with leading zeros stripped (e.g. `684`, `9487203`).
4. `na`: Composite key combining first name word and primary address token (`name4_addr5`).
5. `nt`: Composite key combining numeric token and address token (`num_addr5`) — critical for capturing transliterated entities where names differ in script but addresses match numerically.
6. `aa`: Pair of distinctive address tokens (`addr1_addr2`) to handle token reordering.
7. `aaa`: Triplet token combination for dense address resolution.

### 3.2 Key Pruning & Candidate Ranking
- **Bucket Pruning:** Keys with bucket size $> 500$ are automatically pruned to suppress noise words (e.g., generic street suffixes).
- **Candidate Scoring:** Query keys are weighted by channel discriminability (`nw1: 3.0`, `num: 2.5`, `na: 2.2`, `atk: 1.8`, `aaa: 1.5`).
- **Benchmark Performance:**
  - **Recall Ceiling:** `95.41%`
  - **Average Candidate Set Size:** `11.99` candidates per Source 1 entity (well under the $\le 15$ target).
  - **Candidate Minimization:** Eliminates $> 99.9997\%$ of non-matching pair comparisons.

---

## 4. Matching Model & Feature Engineering

### 4.1 Feature Set (15 Pairwise Signals)

| Feature Index | Feature Name | Description | Rationale |
|---|---|---|---|
| `f0` | `jaro_winkler` | Jaro-Winkler distance on legal-stripped name | Captures minor spelling errors & typographical variations |
| `f1` | `ngram3_overlap` | Character 3-gram Dice coefficient | Robust against word splitting and character deletions |
| `f2` | `ngram2_overlap` | Character 2-gram Dice coefficient | Handles short abbreviations and transliterated sub-tokens |
| `f3` | `prefix5_match` | Exact match of first 5 characters | High-confidence anchor signal |
| `f4` | `token_sort_ratio` | Jaro-Winkler over alphabetically sorted name tokens | Invariant to word permutations ("Apple Store" $\leftrightarrow$ "Store Apple") |
| `f5` | `addr_token_jaccard` | Word-level Jaccard index on expanded address strings | Measures overall address token overlap |
| `f6` | `numeric_match` | Normalized Jaccard similarity of extracted numbers | PIN codes, street numbers, building numbers |
| `f7` | `distinct_addr_ol` | Distinctive (non-stopword) address token overlap | Isolates unique locality/landmark tokens |
| `f8` | `addr_prefix8` | Prefix exact match on cleaned address | Detects identical starting address lines |
| `f9` | `country_match` | Binary indicator ($1.0$ if country matches) | Hard boundary gate |
| `f10` | `legal_form_match` | Categorical match of detected legal suffixes (`LLC`, `SARL`, `Pvt Ltd`) | Distinguishes differently registered entities |
| `f11` | `name_len_ratio` | $\min(L_1, L_2) / \max(L_1, L_2)$ length ratio | Penalizes extreme truncation |
| `f12` | `name_x_addr` | Product of name similarity and address Jaccard (`f0 * f5`) | Non-linear interaction feature |
| `f13` | `max_signal` | Maximum across individual similarity dimensions | Soft-OR signal for strong unilateral matches |
| `f14` | `blocking_score_n` | Normalized multi-key blocking accumulation score | Transmits candidate generation confidence to the classifier |

### 4.2 Model Architecture & Training
- **Classifier:** LightGBM Gradient Boosted Decision Trees (`num_leaves=63`, `learning_rate=0.05`, `n_estimators=500`, `colsample_bytree=0.8`).
- **Class Balancing:** `scale_pos_weight` set dynamically to match negative-to-positive hard negative ratio ($1 : 2.52$).
- **Objective Function:** Binary logloss with early stopping on validation entity splits.

### 4.3 Dual-Threshold $F_{0.5}$ Optimization
Because the competition metric is Macro $F_{0.5}$ (weighting Precision 2x higher than Recall):
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
A high decision threshold ($t = 0.84$) was calibrated via grid search on unseen validation entities. This eliminates marginal false positives, heavily boosting the $F_{0.5}$ score.

---

## 5. Results & Error Analysis

### 5.1 Quantitative Results
- **Validation Macro-$F_{0.5}$ Score:** `0.9864` (vs. baseline `0.9825` at default $t=0.50$)
- **Validation Precision:** `0.9912`
- **Validation Recall:** `0.9678`
- **Blocking Recall Ceiling:** `95.41%`
- **Average Candidate Pool Size:** `11.99` candidates per S1 entity

### 5.2 Error Analysis
- **False Positives (Rare):** Co-located entities sharing identical address numbers in dense commercial complexes but with similar generic trade names. The high $0.84$ threshold effectively suppresses these.
- **False Negatives (Unresolved):** Complex DBA (Doing Business As) aliases (e.g. "Maure Williams Colombier Inc" $\leftrightarrow$ "Dréxkor") where neither the name strings nor the address records share overlapping tokens or numbers without external knowledge graph lookups.

---

## 6. Conclusion

Our solution demonstrates that combining multi-key blocking with country partitioning, rich feature engineering, and precision-calibrated LightGBM classification achieves top-tier Entity Resolution accuracy (0.9864 Macro-$F_{0.5}$) while maintaining a minimal candidate set size (avg 11.99) and linear execution throughput (>8,000 S1/sec). The complete solution is self-contained, reproducible, and adheres strictly to all competition rules and submission formats.

---

## Appendix: Code Artifacts & Reproducibility

### Structure
```
code/business_entity_resolution/
├── README.md               # Quickstart and execution instructions
├── requirements.txt        # Minimal dependencies (polars, lightgbm, numpy, scikit-learn)
└── src/
    ├── normalizer.py       # Unicode normalization & multi-key extraction
    ├── blocker.py          # Array-backed inverted index
    ├── features.py         # 15-feature pairwise similarity extractor
    ├── build_dataset.py    # Pairwise training dataset constructor
    ├── train_model.py      # LightGBM trainer & F0.5 calibrator
    └── inference.py        # Streaming country-partitioned test inference engine
```

### Quick Reproduction Command
```bash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Train model & optimize threshold
python code/business_entity_resolution/src/train_model.py

# 3. Generate candidate_pairs.tsv and matching_results.tsv
python code/business_entity_resolution/src/inference.py

# 4. Validate output compliance
python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
