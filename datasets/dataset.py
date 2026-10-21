import os
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Dataset, Data

from pymatgen.core.structure import Structure

from config import CSV_PATH, CIF_DIR
from features.graph_builder import structure_to_graph, augment_structure

# ============================================================
# MAIN DATASET
# ============================================================

class SolidElectrolyteDataset(Dataset):

    _STRUCTURE_CACHE = {}  # Class-level cache for Structure objects

    def __init__(
        self,
        csv_path=CSV_PATH,
        cif_dir=CIF_DIR,
        indices=None,
        scaler=None,
        fit_scaler=True,
        cache_graph=True,
        mode="multiview",  # multiview / graph / descriptor
        verbose=True,
        aug_method=None,   # None / "rattle" / "symmetry" / "both"
        aug_factor=0       # Number of augmented samples per original sample
    ):
        """
        Parameters
        ----------
        indices : subset indices
        scaler : train scaler
        fit_scaler : only train set=True
        mode :
            multiview
            graph
            descriptor
        aug_method : "rattle", "symmetry", or "both"
        aug_factor : if > 0, expand the dataset
        """

        super().__init__()

        self.csv_path = csv_path
        self.cif_dir = cif_dir
        self.cache_graph = cache_graph
        self.mode = mode
        self.verbose = verbose
        self.aug_method = aug_method
        self.aug_factor = int(aug_factor)

        # ----------------------------
        # load dataframe
        # ----------------------------
        self.df = pd.read_csv(csv_path)

        # ----------------------------
        # descriptor columns
        # ----------------------------
        self.desc_cols = [
            "density",
            "band_gap",
            "lattice_a",
            "lattice_b",
            "lattice_c",
            "avg_eneg",
            "ionic_radius",
            "activation_energy"
        ]

        # ----------------------------
        # missing value process
        # ----------------------------
        self.df[self.desc_cols] = self.df[self.desc_cols].fillna(
            self.df[self.desc_cols].median()
        )

        # ----------------------------
        # subset split
        # ----------------------------
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        # ----------------------------
        # scaler
        # ----------------------------
        if scaler is None:
            scaler = StandardScaler()

        self.scaler = scaler

        if fit_scaler:
            self.df[self.desc_cols] = self.scaler.fit_transform(
                self.df[self.desc_cols]
            )
        else:
            self.df[self.desc_cols] = self.scaler.transform(
                self.df[self.desc_cols]
            )

        # ----------------------------
        # cache dir
        # ----------------------------
        self.cache_dir = os.path.join(
            os.path.dirname(csv_path),
            "graph_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        if verbose:
            print("=" * 60)
            print("SolidElectrolyteDataset Loaded")
            print("Rows:", len(self.df))
            print("Mode:", self.mode)
            print("=" * 60)

    # ========================================================
    # Basic
    # ========================================================

    def len(self):
        return len(self.df) * (1 + self.aug_factor)

    # ========================================================
    # Graph Cache
    # ========================================================

    def graph_cache_path(self, material_id):
        return os.path.join(
            self.cache_dir,
            f"{material_id}.pkl"
        )

    def load_graph(self, material_id):

        cache_path = self.graph_cache_path(material_id)

        if self.cache_graph and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Use cache for Structure objects
        if material_id in self._STRUCTURE_CACHE:
            structure = self._STRUCTURE_CACHE[material_id]
        else:
            cif_path = os.path.join(
                self.cif_dir,
                material_id + ".cif"
            )
            structure = Structure.from_file(cif_path)
            self._STRUCTURE_CACHE[material_id] = structure

        graph = structure_to_graph(structure)

        if self.cache_graph:
            with open(cache_path, "wb") as f:
                pickle.dump(graph, f)

        return graph

    # ========================================================
    # Get item
    # ========================================================

    def get(self, idx):

        original_len = len(self.df)
        is_augmented = idx >= original_len
        original_idx = idx % original_len

        row = self.df.iloc[original_idx]

        material_id = row["material_id"]

        y = np.log10(
            float(row["conductivity"]) + 1e-8
        )

        desc = row[self.desc_cols].values.astype(float)

        # ----------------------------------------------------
        # graph only / multiview
        # ----------------------------------------------------
        if self.mode in ["multiview", "graph"]:

            if is_augmented and self.aug_method:
                # Online Augmentation (No Cache for result, but use cache for source structure)
                if material_id in self._STRUCTURE_CACHE:
                    structure = self._STRUCTURE_CACHE[material_id]
                else:
                    cif_path = os.path.join(
                        self.cif_dir,
                        material_id + ".cif"
                    )
                    structure = Structure.from_file(cif_path)
                    self._STRUCTURE_CACHE[material_id] = structure

                method = self.aug_method
                if method == "both":
                    import random
                    method = random.choice(["rattle", "symmetry"])

                aug_struct = augment_structure(structure, method=method)
                x, edge_index, edge_attr = structure_to_graph(aug_struct)
            else:
                x, edge_index, edge_attr = self.load_graph(
                    material_id
                )
            
            # Debug: print shape of x for the first sample
            if idx == 0 and self.verbose:
                print(f"DEBUG: Loaded graph for {material_id}, x.shape: {x.shape}")

        else:
            # descriptor only mode dummy graph
            x = torch.zeros((1, 8), dtype=torch.float)
            edge_index = torch.tensor(
                [[0], [0]],
                dtype=torch.long
            )
            edge_attr = torch.zeros((1, 16))

        # ----------------------------------------------------
        # descriptor only / multiview
        # ----------------------------------------------------
        if self.mode in ["multiview", "descriptor"]:

            desc_tensor = torch.tensor(
                desc,
                dtype=torch.float
            ).view(1, -1)

        else:
            desc_tensor = torch.zeros((1, len(self.desc_cols)))

        # ----------------------------------------------------
        # build Data
        # ----------------------------------------------------
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(
                [[y]],
                dtype=torch.float
            ),
            desc=desc_tensor,
            material_id=str(material_id)
        )

        return data

# ============================================================
# Helper Functions
# ============================================================

def build_train_test_datasets(
    train_idx,
    test_idx,
    csv_path=CSV_PATH,
    cif_dir=CIF_DIR,
    mode="multiview",
    aug_method=None,
    aug_factor=0
):
    """
    防泄漏标准构建方式
    """

    full_df = pd.read_csv(csv_path)

    desc_cols = [
        "density",
        "band_gap",
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "avg_eneg",
        "ionic_radius",
        "activation_energy"
    ]

    scaler = StandardScaler()

    scaler.fit(
        full_df.iloc[train_idx][desc_cols]
    )

    train_set = SolidElectrolyteDataset(
        csv_path=csv_path,
        cif_dir=cif_dir,
        indices=train_idx,
        scaler=scaler,
        fit_scaler=False,
        mode=mode,
        aug_method=aug_method,
        aug_factor=aug_factor
    )

    test_set = SolidElectrolyteDataset(
        csv_path=csv_path,
        cif_dir=cif_dir,
        indices=test_idx,
        scaler=scaler,
        fit_scaler=False,
        mode=mode,
        verbose=False
    )

    return train_set, test_set


# ============================================================
# KFold helper
# ============================================================

def build_kfold_dataset(
    train_idx,
    val_idx,
    csv_path=CSV_PATH,
    mode="multiview"
):
    return build_train_test_datasets(
        train_idx,
        val_idx,
        csv_path=csv_path,
        mode=mode
    )