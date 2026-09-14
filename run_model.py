"""Train and evaluate KAGNet for a single cancer type using repeated
stratified k-fold cross-validation.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from model import KAGNetModel
from utils import (build_dataset, compute_metrics, make_train_val_split,
                    positive_class_weight, seed_everything)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate KAGNet on one TCGA cancer type.")

    parser.add_argument("--data_root", type=str, default="Data",
                         help="Path to the Data/ directory (see README).")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Directory where checkpoints, scores, and metrics are written.")
    parser.add_argument("--target", type=str, required=True,
                         help="Cancer type to run, e.g. BRCA, BLCA, LUAD, LIHC, THCA, "
                              "LUSC, ESCA, PRAD, STAD, COAD, UCEC, CESC.")
    parser.add_argument("--negative_fname", type=str, default="2187false.txt",
                         help="Filename (under Data/labels/) of the shared passenger-gene list.")

    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--cls_hidden", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=32,
                         help="Projection dimension matching the AlphaGenome pretraining input space.")
    parser.add_argument("--dropout", type=float, default=0.4,
                         help="Dropout applied both inside each GCN encoder (between its "
                              "two layers) and inside each MLP classifier.")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min_epoch_select", type=int, default=50)
    parser.add_argument("--alpha_branch", type=float, default=1.0,
                         help="Fixed weight of the averaged branch-level loss added to the fusion loss.")

    parser.add_argument("--device", type=str, default=None,
                         help="'cuda' or 'cpu'; auto-detected if omitted.")

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def train_fold(cfg, dataset, run_id, fold_id, train_idx, val_idx, test_idx, out_dir):
    device = cfg.device
    x_dev = dataset["x_features"].to(device)
    y_dev = dataset["y"].to(device).float()
    y_np = dataset["y"].numpy()
    edge_indices = dataset["edge_indices"]
    edge_attrs = dataset["edge_attrs"]
    network_names = dataset["network_names"]

    pretrain_dir = os.path.join(cfg.data_root, "pretrained_encoders")

    model = KAGNetModel(
        feat_dim=dataset["feat_dim"], hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim,
        network_names=network_names, cls_hidden=cfg.cls_hidden, proj_dim=cfg.proj_dim,
        dropout=cfg.dropout,
    ).to(device)
    model.load_pretrained_encoders(pretrain_dir, verbose=(run_id == 1 and fold_id == 1))

    beta, n_pos, n_neg = positive_class_weight(y_np, train_idx)
    beta_t = torch.tensor(beta, dtype=torch.float, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    y_train = y_dev[train_idx_t]
    weight = torch.where(y_train == 1, beta_t.expand_as(y_train), torch.ones_like(y_train))

    train_labeled = train_idx[y_np[train_idx] != -1]
    val_labeled = val_idx[y_np[val_idx] != -1]
    test_labeled = test_idx[y_np[test_idx] != -1]

    best_state, best_val_auprc, best_epoch, wait = None, -1.0, -1, 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        branch_probs = model(x_dev, edge_indices, edge_attrs)  # (N, n_networks)
        p_fusion = branch_probs.mean(dim=1)

        p_fusion_train = torch.clamp(p_fusion[train_idx_t], 1e-6, 1.0 - 1e-6)
        loss_fusion = F.binary_cross_entropy(p_fusion_train, y_train, weight=weight)

        branch_losses = []
        for k in range(branch_probs.size(1)):
            p_k_train = torch.clamp(branch_probs[train_idx_t, k], 1e-6, 1.0 - 1e-6)
            branch_losses.append(F.binary_cross_entropy(p_k_train, y_train, weight=weight))
        loss_branch = torch.stack(branch_losses).mean()

        loss = loss_fusion + cfg.alpha_branch * loss_branch
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_probs = model(x_dev, edge_indices, edge_attrs)
            p_fusion_eval = eval_probs.mean(dim=1).cpu().numpy()

        tr_m = compute_metrics(y_np[train_labeled], p_fusion_eval[train_labeled])
        va_m = compute_metrics(y_np[val_labeled], p_fusion_eval[val_labeled])
        te_m = compute_metrics(y_np[test_labeled], p_fusion_eval[test_labeled])  # monitoring only

        history.append({
            "epoch": epoch, "loss_total": float(loss.item()),
            "loss_fusion": float(loss_fusion.item()), "loss_branch": float(loss_branch.item()),
            "train_auprc": tr_m["auprc"], "val_auprc": va_m["auprc"], "test_auprc": te_m["auprc"],
        })

        if epoch >= cfg.min_epoch_select and va_m["auprc"] > best_val_auprc:
            best_val_auprc, best_epoch, wait = va_m["auprc"], epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch >= cfg.min_epoch_select:
            wait += 1

        if epoch == 1 or epoch % 25 == 0:
            print(f"  [run{run_id:02d} fold{fold_id:02d} epoch{epoch:03d}] "
                  f"loss={loss.item():.4f} train={tr_m['auprc']:.4f} "
                  f"val={va_m['auprc']:.4f} test={te_m['auprc']:.4f}")

        if wait >= cfg.patience:
            print(f"  Early stop at epoch {epoch} (best epoch {best_epoch}, "
                  f"best val AUPRC {best_val_auprc:.4f})")
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint was selected; lower --min_epoch_select "
                            "or increase --max_epochs.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_probs = model(x_dev, edge_indices, edge_attrs)
        prob_per_net = final_probs.cpu().numpy()
        prob_fusion = final_probs.mean(dim=1).cpu().numpy()

    te_final = compute_metrics(y_np[test_labeled], prob_fusion[test_labeled])
    va_final = compute_metrics(y_np[val_labeled], prob_fusion[val_labeled])

    result = dict(
        run=run_id, fold=fold_id, best_epoch=best_epoch, n_pos=n_pos, n_neg=n_neg,
        val_auprc=va_final["auprc"], val_auroc=va_final["auroc"],
        test_auprc=te_final["auprc"], test_auroc=te_final["auroc"],
    )

    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, "best_model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    pd.DataFrame([result]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    pd.DataFrame({
        "gene": dataset["global_genes"], "score_fusion": prob_fusion, "label": y_np,
        **{f"score_{n}": prob_per_net[:, k] for k, n in enumerate(network_names)},
    }).to_csv(os.path.join(out_dir, "scores.csv"), index=False)

    return result


def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Device: {cfg.device}")
    print(f"Data root: {cfg.data_root}")
    print(f"Target cancer: {cfg.target}")
    print(f"Output dir: {cfg.output_dir}")

    dataset = build_dataset(cfg.data_root, cfg.target, cfg.device, cfg.negative_fname)

    out_dir = os.path.join(cfg.output_dir, cfg.target)
    os.makedirs(out_dir, exist_ok=True)

    y_np = dataset["y"].numpy()
    labeled_idx = np.where(y_np != -1)[0]
    y_labeled = y_np[labeled_idx]

    all_results, run_means = [], []

    for run in range(cfg.n_runs):
        seed = cfg.seed + run
        seed_everything(seed)
        print(f"\n{'=' * 80}\n[{cfg.target}] run {run + 1}/{cfg.n_runs} (seed={seed})\n{'=' * 80}")

        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=seed)
        fold_scores = []

        for fold, (tv_rel, te_rel) in enumerate(skf.split(labeled_idx, y_labeled), start=1):
            trainval_idx = labeled_idx[tv_rel]
            test_idx = labeled_idx[te_rel]
            train_idx, val_idx = make_train_val_split(trainval_idx, y_np[trainval_idx], seed + fold)

            seed_everything(seed + fold)
            fold_dir = os.path.join(out_dir, f"run_{run + 1:02d}_fold_{fold:02d}")
            result = train_fold(cfg, dataset, run + 1, fold, train_idx, val_idx, test_idx, fold_dir)
            all_results.append(result)
            fold_scores.append(result["test_auprc"])
            print(f"  [run{run + 1} fold{fold}] val_AUPRC={result['val_auprc']:.4f} "
                  f"test_AUPRC={result['test_auprc']:.4f}")

        run_means.append(float(np.nanmean(fold_scores)))
        print(f"[{cfg.target}] run {run + 1} mean test AUPRC = {run_means[-1]:.4f}")

    pd.DataFrame(all_results).to_csv(
        os.path.join(cfg.output_dir, f"results_{cfg.target}_all_folds.csv"), index=False)

    rm = np.array(run_means)
    summary = {
        "cancer": cfg.target,
        "gene_universe": len(dataset["global_genes"]),
        "feat_dim": dataset["feat_dim"],
        "n_runs": cfg.n_runs, "n_folds": cfg.n_folds,
        "network_names": dataset["network_names"],
        "run_means": run_means,
        "final_auprc_mean": float(np.nanmean(rm)),
        "final_auprc_std": float(np.nanstd(rm)),
    }
    with open(os.path.join(cfg.output_dir, f"summary_{cfg.target}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{cfg.target}] FINAL: AUPRC = {summary['final_auprc_mean']:.4f} "
          f"± {summary['final_auprc_std']:.4f}")


if __name__ == "__main__":
    main()
