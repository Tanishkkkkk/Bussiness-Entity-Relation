# Amazon ML Challenge 2026: Business Entity Resolution

## What This Is
A scalable, high-precision Entity Resolution (ER) pipeline designed to resolve noisy business identity records from three independent data sources (Source 1 deduplicated reference, Source 2, and Source 3). The pipeline takes raw, unaligned records with abbreviations, typos, legal suffix inconsistencies, missing fields, and script transliterations, then identifies all corresponding Source 2/Source 3 entities matching each Source 1 reference record.

## Core Value
Maximizing the competition's macro-averaged **F_0.5 score** (which weights precision 2x higher than recall) while generating a minimal, high-recall candidate set per Source 1 entity that satisfies the candidate ranking evaluation criteria and scales cleanly across 12+ million records.

## Problem & Competition Context
- **Data Scale**:
  - Training: 2,206,821 Source 1, 5,034,616 Source 2, 5,285,603 Source 3 (Total ~12.5M records).
  - Test: 1,732,544 Source 1, 4,887,273 Source 2, 5,082,316 Source 3 (Total ~11.7M records).
  - Unblocked comparison space: ~17.3 trillion pairs.
- **Evaluation Metric**:
  - Macro F_0.5 score computed per Source 1 entity and averaged across all test entities:
    $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  - **Singletons**: ~5.6% of Source 1 entities have 0 matches in Ground Truth. Correctly predicting an empty list awards a perfect 1.0; a single false positive on a singleton crashes the score to 0.0.
  - **Candidate Set Evaluation**: Final ranking heavily weighs the efficiency/compactness of `candidate_pairs.tsv`. A smaller average candidate pool size per Source 1 entity (with high recall ceiling) is rewarded.
- **Open-Set Geographic Scope**:
  - Training covers `US` and `India`.
  - Test set introduces a third country, `France` (259k S1, 703k S2, 731k S3). Country is an open set string label; zero hard-coded country logic is allowed.
- **Deliverables**:
  - `output/matching_results.tsv`: Final entity matches (`source1_entity_id \t matched_entity_ids`).
  - `output/candidate_pairs.tsv`: Blocking candidate set fed to the classifier (`source1_entity_id \t candidate_entity_ids`).
  - `code/business_entity_resolution/`: Reproducible source (`src/`, `README.md`, `requirements.txt`).
  - `Documentation_template.md`: Completed technical methodology report.
  - Final submission zip archive: `<team_name>_submission.zip`.

## Key Architectural Decisions & Strategies

| Decision | Rationale | Status |
| :--- | :--- | :--- |
| **Strict Country Partitioning** | Verified on ground truth that 0.0000% of matches cross country boundaries. Grouping by country slices the search space cleanly with zero recall loss. | Validated |
| **Two-Stage Multi-Key Blocking** | Combines normalized token prefixes, inverted BM25/TF-IDF character n-gram index, and postal/city locality keys to achieve $\ge 97\%$ recall with an average candidate pool of only 8-15 records per S1 entity. | Active |
| **Cross-Lingual & Script Normalization** | Handles transliteration between Latin, Devanagari (Hindi), and Tamil in India, as well as accented French characters (é, è, ê, etc.) via Unicode NFKD normalization and phonetic/token n-gram overlap. | Active |
| **Precision-Biased Matching Classifier (LightGBM)** | Pairwise gradient boosted decision tree model trained on candidate pairs with comprehensive string, address token sort/set, numeric/ZIP token overlap, and candidate margin features. | Active |
| **Dual-Threshold Optimization** | Uses validation set grid search to tune an entity-level singleton threshold (rejecting candidates when max probability is low) and pairwise match threshold to directly maximize Macro F_0.5. | Active |

## Hard Constraints
- **Strictly Prohibited**: No external lookups, APIs, geocoders, or web queries. Closed-world problem using only provided dataset.
- **Model Size & License**: <= 8B parameters, MIT/Apache 2.0 open-source license.
- **Format Integrity**: Validated with `utils/validate_submission.py` to ensure exit code 0.

---
*Created: 2026-09-26 | Antigravity GSD Initial Plan*
