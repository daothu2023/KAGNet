"""KAGNet model: six independent GCN branches, each with its own projection,
AlphaGenome-pretrained encoder, and MLP classifier, fused by averaging
branch-level probabilities.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.4, self_loops=True):
        super().__init__()
        # cached=True: safe since a new encoder is created for every fold.
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=self_loops,
                              normalize=True, cached=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=self_loops,
                              normalize=True, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        return h


class KAGNetModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim, out_dim, network_names, cls_hidden,
                 proj_dim, dropout=0.4, use_edge_weight=True, self_loops=True):
        super().__init__()
        self.network_names = network_names
        self.n_networks = len(network_names)
        self.use_edge_weight = use_edge_weight

        self.projections = nn.ModuleList([
            nn.Linear(feat_dim, proj_dim) for _ in range(self.n_networks)
        ])
        self.encoders = nn.ModuleList([
            GCNEncoder(proj_dim, hidden_dim, out_dim, dropout=dropout, self_loops=self_loops)
            for _ in range(self.n_networks)
        ])

        residual_dim = out_dim + feat_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(residual_dim, cls_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(cls_hidden, 1),
            )
            for _ in range(self.n_networks)
        ])

    def load_pretrained_encoders(self, pretrain_dir: str, verbose=False):
        """Load the AlphaGenome-pretrained GCN weights into each branch's encoder."""
        for k, net_name in enumerate(self.network_names):
            path = os.path.join(pretrain_dir, f"ag_encoder_{net_name}.pt")
            state = torch.load(path, map_location="cpu")
            missing, _ = self.encoders[k].load_state_dict(state, strict=False)
            if verbose:
                msg = f"  [branch {k}={net_name}] loaded {path}"
                if missing:
                    msg += f" (missing keys: {missing})"
                print(msg)

    def forward(self, x_features, edge_indices, edge_attrs):
        branch_probs = []
        for k in range(self.n_networks):
            edge_index = edge_indices[k]
            edge_weight = edge_attrs[k] if self.use_edge_weight else None

            x_proj = self.projections[k](x_features)
            h = self.encoders[k](x_proj, edge_index, edge_weight)
            r = torch.cat([h, x_features], dim=1)
            logit = self.heads[k](r).squeeze(-1)
            branch_probs.append(torch.sigmoid(logit))

        return torch.stack(branch_probs, dim=1)
