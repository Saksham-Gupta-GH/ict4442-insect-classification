"""
Dataset Acquisition & Subset Curation Utility.
Supports InsectSet459 and InsectSet66 curation into the working subset (species with >= 40 samples).
"""

import os
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np


class InsectSubsetCurator:
    """
    Filters and stratifies insect acoustic datasets.
    """

    def __init__(self, min_samples_per_species: int = 40):
        self.min_samples_per_species = min_samples_per_species

    def curate_working_subset(
        self,
        metadata_df: pd.DataFrame,
        species_col: str = "species",
        file_col: str = "file_path",
        temp_col: Optional[str] = "temperature"
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Filters species with at least `min_samples_per_species` recordings
        and creates integer label mappings.
        """
        # Count per species
        counts = metadata_df[species_col].value_counts()
        valid_species = counts[counts >= self.min_samples_per_species].index.tolist()

        filtered_df = metadata_df[metadata_df[species_col].isin(valid_species)].copy()

        # Create species-to-id mapping
        species_to_id = {sp: idx for idx, sp in enumerate(sorted(valid_species))}
        filtered_df["label"] = filtered_df[species_col].map(species_to_id)

        print(f"Original species: {len(counts)}, Retained species (>= {self.min_samples_per_species} recordings): {len(valid_species)}")
        print(f"Total curated audio recordings: {len(filtered_df)}")

        return filtered_df, species_to_id

    def generate_stratified_splits(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Assigns each recording to 'train', 'val', or 'test' split via stratified sampling.
        """
        df = df.copy()
        df["split"] = ""

        np.random.seed(random_state)

        for label, group in df.groupby(label_col):
            indices = group.index.values
            np.random.shuffle(indices)

            n = len(indices)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

            df.loc[train_idx, "split"] = "train"
            df.loc[val_idx, "split"] = "val"
            df.loc[test_idx, "split"] = "test"

        return df
