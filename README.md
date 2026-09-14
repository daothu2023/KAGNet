# KAGNet: Knowledge-Enriched Multi-PPI Learning for Cancer Driver Gene Prioritization

This repository contains the training pipeline for **KAGNet**, described in
"KAGNet: Knowledge-Enriched Multi-PPI Learning for Cancer Driver Gene
Prioritization". KAGNet combines:

- AlphaGenome-guided pretraining of six PPI-specific GCN encoders,
- knowledge-graph-derived gene embeddings from a heterogeneous graph
  transformer (HGT) trained on PrimeKG,
- and prediction-level integration across six PPI networks (CPDB, STRING,
  MultiNet, PCNet, IRefIndex, IRefIndex2015), jointly optimized with a
  fusion loss and per-branch auxiliary losses.

## Repository structure

```
KAGNet/
├── README.md
├── requirements.txt
├── model.py          # GCN encoder + multi-branch KAGNet model
├── utils.py           # data loading, seeding, metrics, class weighting
├── run_model.py        # training / cross-validation for one cancer type
└── Data/                # dataset (already included in this repository)
    ├── PPI/
    ├── labels/
    ├── node_features/
    │   ├── kg_features/
    │   └── multiomics_features/
    └── pretrained_encoders/
```

## Data layout

```
Data/
├── PPI/
│   ├── CPDB_PPI.csv
│   ├── STRINGdb_PPI.csv
│   ├── MULTINET_PPI.csv
│   ├── PCNET_PPI.csv
│   ├── IREF_PPI.csv
│   └── IREF_2015_PPI.csv
├── labels/
│   ├── 2187false.txt                    # shared passenger (negative) gene list
│   └── {CANCER}true.txt                 # e.g. BRCAtrue.txt, one file per cancer type
├── node_features/
│   ├── kg_features/
│   │   ├── kg_embeddings_{CANCER}.npy   # (N_gene, 16) HGT embeddings, one per cancer type
│   │   └── kg_gene_names.txt            # gene symbols, shared across cancer types,
│   │                                     # row-aligned with every kg_embeddings_*.npy
│   └── multiomics_features/
│       └── multiomics_features_{CANCER}.csv
└── pretrained_encoders/
    └── ag_encoder_{NETWORK}.pt          # AlphaGenome-pretrained GCN weights, one per network
```

Each PPI CSV has columns `u, v, weight` (gene symbols and an edge weight).

`{CANCER}` refers to one of the 12 TCGA cancer types used in the paper:
`BRCA, BLCA, LUAD, LIHC, THCA, LUSC, ESCA, PRAD, STAD, COAD, UCEC, CESC`.

`{NETWORK}` refers to one of the six PPI networks, using the same internal
names as above: `CPDB, STRING, MULTINET, PCNet, IRefIndex, IRefIndex_2015`.

## Requirements

- Python 3.9+
- torch >= 2.0
- torch-geometric >= 2.4
- numpy >= 1.24
- pandas >= 2.0
- scikit-learn >= 1.3

Install with:

```bash
git clone https://github.com/daothu2023/KAGNet.git
cd KAGNet
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; the pipeline also runs on CPU, just
considerably slower — the quick sanity check below can take several
minutes on CPU for a single cancer type.

## Running

`run_model.py` trains and evaluates KAGNet for **one cancer type per run**:

```bash
python run_model.py --data_root Data --output_dir outputs --target BRCA
```

To run all 12 cancer types, call the script once per cancer type, e.g. from
a shell loop:

```bash
for c in BRCA BLCA LUAD LIHC THCA LUSC ESCA PRAD STAD COAD UCEC CESC; do
    python run_model.py --data_root Data --output_dir outputs --target "$c"
done
```

Key hyperparameters (defaults match the paper) can be overridden from the
command line, e.g.:

```bash
python run_model.py --data_root Data --output_dir outputs --target BRCA \
    --n_runs 10 --n_folds 5 --dropout 0.4 --lr 1e-3 --weight_decay 5e-4 \
    --patience 80 --min_epoch_select 50 --alpha_branch 1.0
```

For a quick sanity check of the pipeline (a few epochs, a single run) before
committing to the full 10×5 cross-validation:

```bash
python run_model.py --data_root Data --output_dir outputs --target BRCA \
    --n_runs 1 --n_folds 2 --max_epochs 10 --min_epoch_select 2 --patience 5
```

Omit these flags to reproduce the AUPRC values reported in the paper.

Run `python run_model.py --help` for the full list of options.

## Outputs

For each `(run, fold)` combination, the pipeline writes under
`{output_dir}/{target}/run_XX_fold_YY/`:

- `best_model.pt` — model weights at the best validation-AUPRC epoch,
- `history.csv` — per-epoch training/validation/test AUPRC and loss curves,
- `metrics.csv` — final validation/test AUPRC and AUROC for that fold,
- `scores.csv` — fused and per-branch predicted probabilities for every gene.

Once all runs/folds for a cancer type complete, `results_{target}_all_folds.csv`
and `summary_{target}.json` are written directly under `{output_dir}`,
summarizing the mean ± std test AUPRC across the 10×5 cross-validation runs.
