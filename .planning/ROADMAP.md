# Roadmap: Amazon ML Challenge 2026 - Business Entity Resolution

## Phases Overview

```
Phase 1: Ingestion & Normalization ──► Phase 2: Scalable Blocking Engine ──► Phase 3: Feature Pipeline
                                                                                   │
Phase 6: Final Packaging & Docs  ◄── Phase 5: Test Inference & Validation ◄── Phase 4: Model Training & F_0.5 Tuning
```

---

### Phase 1: Data Ingestion & Preprocessing Foundation
- **Goal**: High-speed, low-memory ingestion and cleaning of multi-source business records (handling English, Indic transliterations, and French conventions).
- **Deliverables**:
  - Polars/DuckDB streaming reader for all source TSVs.
  - Name normalizer (Unicode NFKD, lowercase, punctuation cleanup, legal suffix standardization across US/India/France: `Inc`, `LLC`, `Ltd`, `Pvt Ltd`, `SARL`, `SASU`, `EURL`).
  - Address normalizer (street abbreviations `St`/`Street`/`Rue`/`Blvd`, numeric token preservation, PIN/ZIP code extraction).
- **Validation**: Pass 100% of sample pairs through normalizer and verify token standardization.

### Phase 2: Scalable Candidate Generation (Blocking Engine)
- **Goal**: Slash the $1.73\text{M} \times 10\text{M} \approx 17.3\text{ Trillion}$ comparison space to $\le 15$ candidates per Source 1 entity while preserving $\ge 95\%$ recall.
- **Deliverables**:
  - Open-set country partitioning stream.
  - Dual blocking strategy:
    1. Inverted character n-gram / token index (BM25 / TF-IDF sparse similarity) on core business name tokens.
    2. Locality + core name prefix hashing (blocking on postal/city code + first 2 distinctive name words).
  - Candidate generation script producing top-$K$ candidates per S1 entity.
  - Evaluation of blocking quality on train set: Candidate Pool Size, Reduction Ratio, and Recall Ceiling.
- **Validation**: Achieve $\ge 95\%$ recall on training ground truth with average candidate count $\le 12$ per S1.

### Phase 3: Feature Engineering & Pairwise Dataset Construction
- **Goal**: Vectorized feature extraction capturing subtle identity cues, typos, address reordering, and numeric consistency.
- **Deliverables**:
  - Parallel feature extraction pipeline:
    - Name features: `token_sort_ratio`, `token_set_ratio`, `damerau_levenshtein`, `jaro_winkler`, `first_word_match`, `name_length_ratio`.
    - Address features: `address_token_jaccard`, `numeric_token_match_ratio`, `numeric_token_conflict_flag`, `missing_address_flag`.
    - Structural features: `blocking_rank`, `blocking_score`, `score_margin_to_next`, `source_origin` (`S2` vs `S3`).
  - Labeled dataset generator: Merges true matches (positives) and blocking hard negatives into balanced training chunks.
- **Validation**: Verify feature matrix contains zero NaNs and computes in $<1$ ms per pair batch.

### Phase 4: Model Training & Dual-Threshold F_0.5 Optimization
- **Goal**: High-precision LightGBM classification calibrated specifically for the competition's macro F_0.5 metric.
- **Deliverables**:
  - LightGBM binary classifier training on candidate pairs.
  - Stratified holdout validation split (Source 1 entity-level split to prevent data leakage).
  - Custom Macro F_0.5 evaluation function matching competition specifications exactly.
  - Dual-threshold optimizer:
    - Pairwise matching probability threshold $\theta_{\text{match}}$.
    - Entity-level singleton threshold $\theta_{\text{singleton}}$ (ensuring true singletons remain empty to maximize the 1.0 reward).
- **Validation**: Achieve validation Macro F_0.5 score $\ge 0.85$ on the local evaluation split.

### Phase 5: End-to-End Test Inference & Submission Validation
- **Goal**: Generate final submission TSVs for all 1,732,544 test Source 1 entities and verify against official rules.
- **Deliverables**:
  - Out-of-core streaming inference script generating:
    - `output/candidate_pairs.tsv` (blocking candidate set).
    - `output/matching_results.tsv` (final entity matches).
  - Submission verification runner using `utils/validate_submission.py`.
- **Validation**: `utils/validate_submission.py` outputs `PASS` (exit code 0).

### Phase 6: Code Reproducibility & Methodology Documentation
- **Goal**: Deliver a submission package that wins top rankings in code review and methodology audits.
- **Deliverables**:
  - `code/business_entity_resolution/src/` clean, modular pipeline.
  - `code/business_entity_resolution/README.md` complete reproduction guide.
  - `code/business_entity_resolution/requirements.txt` pinned dependencies.
  - `Documentation_template.md` filled with full architectural details, candidate reduction tables, and error analyses.
  - Final compressed archive: `<team_name>_submission.zip`.
- **Validation**: Zip archive verifies clean extraction and self-contained structure.
