# Requirements: Amazon ML Challenge 2026 - Business Entity Resolution

## System Requirements

### 1. Data Processing & Ingestion
- **REQ-01.1**: Memory-efficient ingestion of 12.5M train records and 11.7M test records using Polars and DuckDB streaming.
- **REQ-01.2**: Comprehensive string and script normalization:
  - Unicode NFKD normalization to strip diacritics/accents (crucial for French and Indian accented text).
  - Normalization and transliteration handling for Devanagari and Tamil scripts in Indian entities.
  - Legal suffix standardizer (Inc, LLC, Corp, Ltd, Pvt Ltd, SARL, SASU, EURL, etc.).
  - Street abbreviation standardizer (Rd/Road, St/Street, Ave/Avenue, Blvd/Boulevard, etc.).
  - Address token reordering and numeric token extraction (house numbers, postal/PIN/ZIP codes).
- **REQ-01.3**: Zero hardcoded country logic; all processing operates dynamically on country labels.

### 2. Candidate Generation (Blocking)
- **REQ-02.1**: Country-level partitioning as an immutable zero-loss blocking boundary.
- **REQ-02.2**: High-recall, low-cardinality multi-key blocking strategy:
  - Inverted token and character n-gram indexing (BM25 / TF-IDF sparse similarity).
  - Locality + name core prefix keys (combining postal/city tokens with distinctive name tokens).
- **REQ-02.3**: Candidate pool constraint: Average candidate count per Source 1 entity strictly constrained to $\le 15$ candidates (to maximize candidate ranking points) while retaining $\ge 95\%$ ground-truth match recall.
- **REQ-02.4**: Generate `output/candidate_pairs.tsv` strictly adhering to format:
  - `source1_entity_id \t candidate_entity_ids` (comma-separated, sorted, test S2/S3 IDs only).
  - All test Source 1 entities present (1,732,544 rows).

### 3. Feature Extraction Pipeline
- **REQ-03.1**: Vectorized string similarities using `rapidfuzz` (Token Sort Ratio, Token Set Ratio, Levenshtein, Jaro-Winkler).
- **REQ-03.2**: Address-specific features:
  - Jaccard token overlap of normalized address tokens.
  - Exact match of numeric tokens (house numbers, postal codes).
  - Penalty for conflicting numeric tokens.
  - Missing address flag indicator.
- **REQ-03.3**: Structural & Ranking features:
  - Blocking score and candidate rank order.
  - Score margin between top-1 and top-2 candidate per Source 1 record.
  - Source origin indicator (`S2` vs `S3`).

### 4. Matching Model & F_0.5 Optimization
- **REQ-04.1**: LightGBM/XGBoost binary classification model with parameter budget $\le 8\text{B}$ parameters and MIT/Apache-2.0 compliance.
- **REQ-04.2**: Representative training pair generation using ground truth positive matches and hard negative candidate pairs sampled from blocking output.
- **REQ-04.3**: Exact Macro F_0.5 metric evaluation implementation:
  $$F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$$
- **REQ-04.4**: Dual-threshold calibration:
  - Pairwise decision threshold $\theta_{\text{match}}$ tuned for precision.
  - Singleton confidence threshold $\theta_{\text{singleton}}$ to prevent false merges on the ~5.6% singleton entities.

### 5. Post-Processing, Validation & Delivery
- **REQ-05.1**: Generate `output/matching_results.tsv` containing all 1,732,544 test Source 1 entities with tab separator and comma-delimited matched IDs.
- **REQ-05.2**: Full validation with `utils/validate_submission.py` ensuring exit code 0 (`PASS`).
- **REQ-05.3**: Build final submission package structured as:
  - `output/matching_results.tsv`
  - `output/candidate_pairs.tsv`
  - `code/business_entity_resolution/src/`
  - `code/business_entity_resolution/README.md`
  - `code/business_entity_resolution/requirements.txt`
  - `Documentation_template.md` (filled methodology writeup)
  - `<team_name>_submission.zip` archive.
