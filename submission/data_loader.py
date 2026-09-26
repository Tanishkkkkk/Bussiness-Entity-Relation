"""
Memory-efficient data loader module using Polars streaming.
Reads TSVs with explicit tab separator, handles missing values, and provides country partitions.
"""

import os
from typing import Dict, Generator, List, Optional, Set, Tuple
import polars as pl


class DataLoader:
    def __init__(self, base_dir: str = "dataset"):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, "train")
        self.test_dir = os.path.join(base_dir, "test")

    def load_source(
        self,
        filepath: str,
        country: Optional[str] = None,
        n_rows: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Load a source TSV file into a Polars DataFrame.
        Always uses explicit separator='\t' to prevent comma-splitting bugs.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Source file not found: {filepath}")

        # Scan as lazyframe for memory efficiency
        lf = pl.scan_csv(
            filepath,
            separator="\t",
            has_header=True,
            null_values=["None", "null", "NaN", "nan", ""],
            infer_schema_length=10000,
        )

        if columns:
            lf = lf.select(columns)

        if country:
            lf = lf.filter(pl.col("country") == country)

        if n_rows:
            lf = lf.limit(n_rows)

        df = lf.collect()

        # Fill nulls in text columns with empty string
        for col in ["business_name", "business_address", "country"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(""))

        return df

    def get_countries(self, filepath: str) -> List[str]:
        """Return unique countries present in a source file without loading full dataset."""
        df = pl.read_csv(filepath, separator="\t", columns=["country"])
        return df["country"].unique().drop_nulls().to_list()

    def load_ground_truth(
        self, filepath: Optional[str] = None, n_rows: Optional[int] = None
    ) -> Dict[str, Set[str]]:
        """
        Load ground truth into a dictionary mapping S1 entity_id -> Set of matched entity_ids.
        Singletons map to an empty set.
        """
        if filepath is None:
            filepath = os.path.join(self.train_dir, "train_ground_truth.tsv")

        lf = pl.scan_csv(
            filepath,
            separator="\t",
            has_header=True,
            null_values=["None", "null", "NaN", "nan"],
        )

        if n_rows:
            lf = lf.limit(n_rows)

        df = lf.collect()

        gt_dict: Dict[str, Set[str]] = {}
        for row in df.iter_rows(named=True):
            s1_id = row["source1_entity_id"]
            matched_str = row["matched_entity_ids"]
            if not matched_str or matched_str.strip() == "":
                gt_dict[s1_id] = set()
            else:
                gt_dict[s1_id] = {m.strip() for m in matched_str.split(",") if m.strip()}

        return gt_dict
