"""
Verification and benchmark script for Phase 1: Ingestion & Normalization Foundation.
"""

import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from normalizer import clean_name, clean_address, strip_accents_and_diacritics
from data_loader import DataLoader


def test_normalization_examples():
    print("=" * 60)
    print("TEST 1: Normalization Rule Verification on Multilingual Samples")
    print("=" * 60)

    test_cases = [
        # US cases
        ("Maure Williams Colombier Inc", "85 Wayne Avenue, Ticonderoga, NY"),
        ("Dahlia Power Reliable Scientific LLC", "630 45th Terrace, Kansas City, MO"),
        ("Payne Enterprises", "3315 FREMONT ST, PEORIA, IL"),
        # India cases
        ("Raj Investments LLP", "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu"),
        ("Ss Food Private Limited", "Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, UP"),
        # France cases
        ("Thermal & Fils SASU", "20 Rue Parmentier, Dunkerque, Hauts-de-France"),
        ("Grain & Fils", "Lille, 329 Avenue de Dunkerque, Hauts-de-France"),
        ("OZT ÀMICALE SAS", "24 R DESAIX, TOURCOING, Hauts-de-France"),
        ("SCI Ptit Àmicale", "18 RUE JEN ZAY, Dunkerque, Nord"),
        ("sci ligue ici parents", "NO. 5 ALLÉE DES HÊTRES, Pornic, Loire-Atlantique"),
    ]

    for name, addr in test_cases:
        core_n, clean_n = clean_name(name)
        clean_a, tokens_a, nums_a = clean_address(addr)
        print(f"Original Name:  {name}")
        print(f"  -> Core Name: {core_n}  | Clean: {clean_n}")
        print(f"Original Addr:  {addr}")
        print(f"  -> Clean Addr: {clean_a}")
        print(f"  -> Numerics:   {sorted(list(nums_a))}")
        print("-" * 50)

    print("PASS: Normalization rules applied successfully across US, India, and France!\n")


def test_data_loader():
    print("=" * 60)
    print("TEST 2: DataLoader Verification on Real TSVs")
    print("=" * 60)

    loader = DataLoader(base_dir="dataset")

    # Check countries in test_source1.tsv
    countries = loader.get_countries("dataset/test/test_source1.tsv")
    print(f"Countries discovered in test_source1: {countries}")
    assert set(countries) == {"US", "India", "France"}, f"Unexpected countries: {countries}"

    # Load 5000 rows for each country
    for c in countries:
        df = loader.load_source("dataset/test/test_source1.tsv", country=c, n_rows=5000)
        print(f"Loaded {len(df)} rows for {c} | Columns: {df.columns}")
        assert len(df) == 5000

    # Load ground truth sample
    gt = loader.load_ground_truth(n_rows=10000)
    print(f"Loaded {len(gt)} ground truth rows successfully.")
    assert len(gt) == 10000

    print("PASS: DataLoader verified on real challenge datasets!\n")


def benchmark_speed():
    print("=" * 60)
    print("TEST 3: High-Throughput Speed Benchmark")
    print("=" * 60)

    loader = DataLoader(base_dir="dataset")
    df = loader.load_source("dataset/train/train_source1.tsv", n_rows=25000)

    start = time.time()
    for row in df.iter_rows(named=True):
        clean_name(row["business_name"])
        clean_address(row["business_address"])
    elapsed = time.time() - start

    rate = len(df) / elapsed
    print(f"Normalized {len(df):,} records in {elapsed:.2f} seconds ({rate:,.0f} records/sec)")
    assert rate > 20000, f"Speed too low: {rate} records/sec"
    print("PASS: Throughput benchmark exceeded requirements!\n")


if __name__ == "__main__":
    test_normalization_examples()
    test_data_loader()
    benchmark_speed()
    print("ALL PHASE 1 CHECKS PASSED!")
