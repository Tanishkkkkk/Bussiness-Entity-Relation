# Amazon ML Challenge 2026: Business Entity Resolution

## Pipeline Overview
This codebase implements an end-to-end entity resolution pipeline across heterogeneous commercial records.

### Directory Structure
```
code/business_entity_resolution/
├── README.md
├── requirements.txt
└── src/
    ├── normalizer.py        # Preprocessing, NFKD normalization, multi-key generation
    ├── blocker.py           # Inverted index with integer indexing & bucket pruning
    ├── features.py          # 15 pairwise similarity feature extractors
    ├── build_dataset.py     # Pairwise dataset construction with hard negative sampling
    ├── train_model.py       # LightGBM classifier training & F0.5 threshold calibration
    ├── inference.py         # Country-partitioned test inference & TSV generator
    ├── test_phase1.py       # Phase 1 unit test & speed benchmark
    └── test_phase2.py       # Phase 2 blocking recall & candidate set benchmark
```

## Setup & Reproduction
```bash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Build feature matrix & train model (Phase 3 & 4)
python code/business_entity_resolution/src/build_dataset.py
python code/business_entity_resolution/src/train_model.py

# 3. Generate test submission files (Phase 5)
python code/business_entity_resolution/src/inference.py
```

Outputs are saved directly to:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`
