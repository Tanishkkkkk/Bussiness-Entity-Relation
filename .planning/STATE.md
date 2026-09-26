# Project State: Amazon ML Challenge 2026 - Business Entity Resolution

## Status Summary
- **Current Phase**: Phase 3: Feature Engineering & Pairwise Dataset Construction
- **Phases Completed**: 2 / 6
- **Last Updated**: 2026-09-26T01:40:30+05:30
- **Progress**: Phase 1 successfully completed and validated. Normalization engine running at >75,000 records/second across US, India, and France. Ingestion engine tested on real TSVs.

## Phase Execution Checklist
- [x] **Phase 1**: Data Ingestion & Preprocessing Foundation (COMPLETED)
  - [x] High-performance `DataLoader` with Polars lazy streaming.
  - [x] Multilingual Unicode NFKD diacritics stripping (French & Indian accents).
  - [x] Multi-country legal suffix standardization (`Inc`, `LLC`, `SARL`, `SASU`, `EURL`, `Pvt Ltd`).
  - [x] Street abbreviation expansion and numeric token extraction.
  - [x] High-throughput verification benchmark (76k+ records/sec).
- [x] **Phase 2**: Scalable Candidate Generation (Blocking Engine) (COMPLETED)
  - [x] Multi-key inverted index: name prefixes (nw1/nw2/nw3), address tokens (atk), numerics (num), combos (na/nt/aa/aaa).
  - [x] Bucket pruning (MAX_BUCKET=400) to prevent noise key explosions.
  - [x] **Recall Ceiling: 95.41%** with **avg 11.99 candidates** per S1 at 8,867 S1/sec.
  - [x] Handles transliteration via address-based keys (num+addr combos).
- [ ] **Phase 3**: Feature Engineering & Pairwise Dataset Construction (IN PROGRESS)
- [ ] **Phase 4**: Model Training & Dual-Threshold F_0.5 Optimization
- [ ] **Phase 5**: End-to-End Test Inference & Submission Validation
- [ ] **Phase 6**: Code Reproducibility & Methodology Documentation

## Key Benchmarks & Metrics
- **Phase 1 Normalization Throughput**: 76,321 records/sec
- **Country Coverage**: US, India, France (open set confirmed)
- **Phase 2 Blocking Recall**: 95.41% (target ≥95%)
- **Phase 2 Avg Candidate Pool**: 11.99 per S1 entity (target ≤15)
- **Phase 2 Query Speed**: 8,867 S1/sec
