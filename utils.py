"""Utility functions for KAGNet: reproducibility, metrics, and data loading.
See README.md for the expected Data/ directory layout.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# Maps the internal network name used throughout the code to the PPI CSV
# filename under Data/PPI/. The same internal name is also used to look up
# the corresponding pretrained encoder: Data/pretrained_encoders/ag_encoder_{name}.pt
NETWORK_FILES = {
    "CPDB": "CPDB_PPI.csv",
    "STRING": "STRINGdb_PPI.csv",
    "MULTINET": "MULTINET_PPI.csv",
    "PCNet": "PCNET_PPI.csv",
    "IRefIndex": "IREF_PPI.csv",
    "IRefIndex_2015": "IREF_2015_PPI.csv",
}


# ----------------------------------------------------------------------
# Reproducibility and metrics
# ----------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out = {}
    try:
        out["auprc"] = average_precision_score(y_true, y_prob)
    except Exception:
        out["auprc"] = np.nan
    try:
        out["auroc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auroc"] = np.nan
    return out


def make_train_val_split(trainval_idx, y_trainval, seed, val_size=0.15):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_rel, va_rel = next(sss.split(trainval_idx, y_trainval))
    return trainval_idx[tr_rel], trainval_idx[va_rel]


def positive_class_weight(y_np, idx, cap=10.0):
    n_pos = int((y_np[idx] == 1).sum())
    n_neg = int((y_np[idx] == 0).sum())
    beta = min(n_neg / max(n_pos, 1), cap)
    return beta, n_pos, n_neg


def zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x - mean) / std


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def read_gene_list(path: str) -> list:
    with open(path) as f:
        return [x.strip() for x in f if x.strip()]


def detect_gene_column(df: pd.DataFrame) -> str:
    for c in ["gene", "Gene", "GENE", "symbol", "SYMBOL",
              "gene_name", "GeneSymbol", "node", "Unnamed: 0"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError("Could not detect a gene-identifier column in the omics table.")


def load_ppi_networks(data_root: str) -> dict:
    ppi_dir = os.path.join(data_root, "PPI")
    ppi_data = {}
    for net_name, fname in NETWORK_FILES.items():
        df = pd.read_csv(os.path.join(ppi_dir, fname))
        c1, c2 = ("u", "v") if {"u", "v"}.issubset(df.columns) else (df.columns[0], df.columns[1])
        df[c1] = df[c1].astype(str)
        df[c2] = df[c2].astype(str)
        if "weight" not in df.columns:
            df["weight"] = 1.0
        ppi_data[net_name] = (df, c1, c2)
        print(f"  [PPI] {net_name}: {len(df)} edges")
    return ppi_data


def load_omics(data_root: str, cancer: str):
    path = os.path.join(data_root, "node_features", "multiomics_features",
                         f"multiomics_features_{cancer}.csv")
    df_omics = pd.read_csv(path)
    gene_col = detect_gene_column(df_omics)
    df_omics[gene_col] = df_omics[gene_col].astype(str).str.strip()
    df_omics = df_omics.drop_duplicates(subset=[gene_col]).reset_index(drop=True)

    feature_cols = [c for c in df_omics.columns if c != gene_col]
    x_raw = (
        df_omics[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values.astype(np.float32)
    )
    x_raw = zscore(x_raw)
    genes = df_omics[gene_col].tolist()
    return genes, x_raw, x_raw.shape[1]


def load_kg_embeddings_aligned(data_root: str, cancer: str, genes_order: list) -> np.ndarray:
    """kg_gene_names.txt is shared across all cancer types (the PrimeKG gene
    set does not change); only kg_embeddings_{cancer}.npy differs per cancer.
    """
    kg_dir = os.path.join(data_root, "node_features", "kg_features")
    emb = np.load(os.path.join(kg_dir, f"kg_embeddings_{cancer}.npy")).astype(np.float32)
    with open(os.path.join(kg_dir, "kg_gene_names.txt")) as f:
        kg_genes = [x.strip() for x in f if x.strip()]

    kg_index = {g: i for i, g in enumerate(kg_genes)}
    dim = emb.shape[1]

    x_kg = np.zeros((len(genes_order), dim), dtype=np.float32)
    missing = np.zeros(len(genes_order), dtype=bool)
    for i, g in enumerate(genes_order):
        if g in kg_index:
            x_kg[i] = emb[kg_index[g]]
        else:
            missing[i] = True

    x_kg = zscore(x_kg)
    x_kg[missing] = 0.0
    print(f"  [KG] {cancer}: dim={dim}, missing={missing.sum()}/{len(genes_order)} genes")
    return x_kg


def build_gene_universe(omics_genes: list, ppi_data: dict) -> list:
    union_ppi_genes = set()
    for _, (df, c1, c2) in ppi_data.items():
        union_ppi_genes |= set(df[c1]) | set(df[c2])
    universe = set(omics_genes) & union_ppi_genes
    return sorted(universe)


def build_edge_indices(ppi_data: dict, gene_to_idx: dict, device: str):
    edge_indices, edge_attrs = [], []
    for net_name, (df, c1, c2) in ppi_data.items():
        mask = df[c1].isin(gene_to_idx) & df[c2].isin(gene_to_idx)
        df_f = df[mask]
        src = df_f[c1].map(gene_to_idx).values
        dst = df_f[c2].map(gene_to_idx).values
        w = df_f["weight"].astype(float).values
        edge_indices.append(torch.tensor(np.vstack([src, dst]), dtype=torch.long).to(device))
        edge_attrs.append(torch.tensor(w, dtype=torch.float).to(device))
        print(f"  [PPI] {net_name}: {len(df_f)} edges in gene universe")
    return edge_indices, edge_attrs


def load_labels(data_root: str, cancer: str, global_genes: list,
                 negative_fname: str = "2187false.txt"):
    labels_dir = os.path.join(data_root, "labels")
    pos_genes = set(read_gene_list(os.path.join(labels_dir, f"{cancer}true.txt")))
    neg_genes = set(read_gene_list(os.path.join(labels_dir, negative_fname)))
    neg_genes = neg_genes - (pos_genes & neg_genes)  # avoid conflicting labels

    y = torch.full((len(global_genes),), -1, dtype=torch.long)
    for i, g in enumerate(global_genes):
        if g in pos_genes:
            y[i] = 1
        elif g in neg_genes:
            y[i] = 0
    return y


def build_dataset(data_root: str, cancer: str, device: str, negative_fname: str = "2187false.txt"):
    """Loads PPI, omics, KG embeddings, and labels for one cancer type."""
    print("\n[INFO] Loading PPI networks ...")
    ppi_data = load_ppi_networks(data_root)

    print(f"\n[INFO] Loading omics features for {cancer} ...")
    omics_genes, x_omics_raw, omics_dim = load_omics(data_root, cancer)
    omics_gene_index = {g: i for i, g in enumerate(omics_genes)}

    global_genes = build_gene_universe(omics_genes, ppi_data)
    gene_to_idx = {g: i for i, g in enumerate(global_genes)}
    n_genes = len(global_genes)
    print(f"\n[INFO] Gene universe: {n_genes} genes | omics_dim={omics_dim}")

    feat_idx = [omics_gene_index[g] for g in global_genes]
    x_omics = torch.tensor(x_omics_raw[feat_idx], dtype=torch.float)

    x_kg_np = load_kg_embeddings_aligned(data_root, cancer, global_genes)
    x_kg = torch.tensor(x_kg_np, dtype=torch.float)

    x_features = torch.cat([x_omics, x_kg], dim=1)
    feat_dim = omics_dim + x_kg.shape[1]
    print(f"[INFO] Combined feature dimension (omics + KG): {feat_dim}")

    y = load_labels(data_root, cancer, global_genes, negative_fname)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"[INFO] Labels: pos={n_pos}, neg={n_neg}, unlabeled={n_genes - n_pos - n_neg}")

    print("\n[INFO] Building per-network edge indices ...")
    edge_indices, edge_attrs = build_edge_indices(ppi_data, gene_to_idx, device)

    return {
        "global_genes": global_genes,
        "x_features": x_features,
        "feat_dim": feat_dim,
        "y": y,
        "edge_indices": edge_indices,
        "edge_attrs": edge_attrs,
        "network_names": list(ppi_data.keys()),
    }
