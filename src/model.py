import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm

from metrics import binary_f1
from utils import mat2bp


class ResConv1DBlock(nn.Module):
    """Residual Conv1d block with channel-wise LayerNorm, GELU, and dropout.

    Operates on tensors shaped (B, C, L). LayerNorm is applied over the channel
    axis only, which keeps positions independent and avoids contaminating
    statistics with padded positions (unlike BatchNorm1d).
    """

    def __init__(self, dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding="same")
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv(x)
        h = h.transpose(1, 2)  # (B, L, C) for LayerNorm over channels
        h = self.norm(h)
        h = h.transpose(1, 2)  # (B, C, L)
        h = self.act(h)
        h = self.drop(h)
        return residual + h


class BindingPredictor(nn.Module):
    """Per-residue binding predictor over [ESM-2 || energy_emb] features.

    Architecture (per residue, then mixed locally by 1D convolutions):
        ESM-2 (1280)  --linear_in-->  (linear_dim)
                                      concat with energy_emb (32)
                                      => hidden_dim = linear_dim + energy_emb_dim
        (B, L, hidden) -> permute -> (B, hidden, L)
        ResConv1DBlock x num_blocks  (residual, LayerNorm, GELU, dropout)
        Conv1d(hidden, 1, kernel_size=1)  -> (B, 1, L) -> squeeze -> logits (B, L)
    """

    def __init__(
        self,
        embed_dim: int,
        linear_dim: int = 32,
        energy_emb_dim: int = 32,
        num_blocks: int = 2,
        kernel_size: int = 9,
        dropout: float = 0.2,
        pos_weight: float = 1.0,
        device: str = "cpu",
        lr: float = 1e-5,
    ):
        super().__init__()
        self.linear_in = nn.Linear(embed_dim, linear_dim)
        hidden_dim = linear_dim + energy_emb_dim
        self.blocks = nn.ModuleList(
            [ResConv1DBlock(hidden_dim, kernel_size=kernel_size, dropout=dropout)
             for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv1d(hidden_dim, 1, kernel_size=1, padding="same")

        self.device = device
        self.register_buffer(
            "pos_weight", torch.tensor([pos_weight], dtype=torch.float32)
        )
        self.threshold = 0.5

        self.to(device)

    def loss_func(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Masked BCE with pos_weight. Padding positions are marked y == -1."""
        mask = y != -1
        return binary_cross_entropy_with_logits(
            yhat[mask], y[mask], pos_weight=self.pos_weight
        )

    def forward(self, x: torch.Tensor, energy_embs: torch.Tensor) -> torch.Tensor:
        # x: (B, L, embed_dim), energy_embs: (B, L, energy_emb_dim)
        x = self.linear_in(x)                       # (B, L, linear_dim)
        x = torch.cat([x, energy_embs], dim=-1)     # (B, L, hidden_dim)
        x = x.transpose(1, 2)                       # (B, hidden_dim, L)
        for block in self.blocks:
            x = block(x)
        x = self.conv_out(x)                        # (B, 1, L)
        return x.squeeze(1)                         # (B, L)

    def fit(self, loader, optimizer):
        self.train()
        loss_acum = 0
        f1_acum = 0
        tn_acum = 0
        fp_acum = 0
        fn_acum = 0
        tp_acum = 0
        zone_tn_acum = 0
        zone_fp_acum = 0
        zone_fn_acum = 0
        zone_tp_acum = 0
        non_zone_tn_acum = 0
        non_zone_fp_acum = 0
        non_zone_fn_acum = 0
        non_zone_tp_acum = 0
        lens = 0
        for batch in tqdm(loader):
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)
            zone_mask = batch[2].to(self.device)
            lens += (y != -1).sum().item()
            energy_embs = batch[5].to(self.device)
            y_pred = self(X, energy_embs)
            loss = self.loss_func(y_pred, y)
            loss_acum += loss.item()
            metrics = binary_f1(y.cpu(), y_pred.detach().cpu(), zone_batch=zone_mask.cpu())
            f1_acum += metrics["f1"]
            tn_acum += metrics["tn"]
            fp_acum += metrics["fp"]
            fn_acum += metrics["fn"]
            tp_acum += metrics["tp"]
            zone_tn_acum += metrics["zone_tn"]
            zone_fp_acum += metrics["zone_fp"]
            zone_fn_acum += metrics["zone_fn"]
            zone_tp_acum += metrics["zone_tp"]
            non_zone_tn_acum += metrics["non_zone_tn"]
            non_zone_fp_acum += metrics["non_zone_fp"]
            non_zone_fn_acum += metrics["non_zone_fn"]
            non_zone_tp_acum += metrics["non_zone_tp"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        assert (tn_acum + fp_acum + fn_acum + tp_acum == lens), "Confusion matrix counts do not sum to total samples"
        assert (zone_tn_acum + zone_fp_acum + zone_fn_acum + zone_tp_acum + non_zone_tn_acum + non_zone_fp_acum + non_zone_fn_acum + non_zone_tp_acum == lens), "Zone confusion counts do not sum to total samples"
        loss_acum /= len(loader)
        f1_acum /= len(loader)
        return {
            "loss": loss_acum,
            "f1": f1_acum,
            "tn": tn_acum,
            "fp": fp_acum,
            "fn": fn_acum,
            "tp": tp_acum,
            "zone_tn": zone_tn_acum,
            "zone_fp": zone_fp_acum,
            "zone_fn": zone_fn_acum,
            "zone_tp": zone_tp_acum,
            "non_zone_tn": non_zone_tn_acum,
            "non_zone_fp": non_zone_fp_acum,
            "non_zone_fn": non_zone_fn_acum,
            "non_zone_tp": non_zone_tp_acum,
        }

    def test(self, loader):
        self.eval()
        loss_acum = 0
        f1_acum = 0
        tn_acum = 0
        fp_acum = 0
        fn_acum = 0
        tp_acum = 0
        zone_tn_acum = 0
        zone_fp_acum = 0
        zone_fn_acum = 0
        zone_tp_acum = 0
        non_zone_tn_acum = 0
        non_zone_fp_acum = 0
        non_zone_fn_acum = 0
        non_zone_tp_acum = 0
        for batch in loader:
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)
            zone_mask = batch[2].to(self.device)
            energy_embs = batch[5].to(self.device)
            with torch.no_grad():
                y_pred = self(X, energy_embs)
                loss = self.loss_func(y_pred, y)
            loss_acum += loss.item()

            metrics = binary_f1(y.cpu(), y_pred.detach().cpu(), zone_batch=zone_mask.cpu())
            f1_acum += metrics["f1"]
            tn_acum += metrics["tn"]
            fp_acum += metrics["fp"]
            fn_acum += metrics["fn"]
            tp_acum += metrics["tp"]
            zone_tn_acum += metrics["zone_tn"]
            zone_fp_acum += metrics["zone_fp"]
            zone_fn_acum += metrics["zone_fn"]
            zone_tp_acum += metrics["zone_tp"]
            non_zone_tn_acum += metrics["non_zone_tn"]
            non_zone_fp_acum += metrics["non_zone_fp"]
            non_zone_fn_acum += metrics["non_zone_fn"]
            non_zone_tp_acum += metrics["non_zone_tp"]
        loss_acum /= len(loader)
        f1_acum /= len(loader)

        return {
            "loss": loss_acum,
            "f1": f1_acum,
            "tn": tn_acum,
            "fp": fp_acum,
            "fn": fn_acum,
            "tp": tp_acum,
            "zone_tn": zone_tn_acum,
            "zone_fp": zone_fp_acum,
            "zone_fn": zone_fn_acum,
            "zone_tp": zone_tp_acum,
            "non_zone_tn": non_zone_tn_acum,
            "non_zone_fp": non_zone_fp_acum,
            "non_zone_fn": non_zone_fn_acum,
            "non_zone_tp": non_zone_tp_acum,
        }

    def collect_scores(self, loader):
        """Returns (y_true, y_prob) flat numpy arrays (padding excluded) for ROC analysis."""
        self.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                X = batch[0].to(self.device)
                y = batch[1]
                energy_embs = batch[5].to(self.device)
                y_pred = self(X, energy_embs).cpu()
                mask = y != -1
                all_labels.append(y[mask].numpy())
                all_probs.append(torch.sigmoid(y_pred[mask]).numpy())
        return np.concatenate(all_labels), np.concatenate(all_probs)

    def pred(self, loader):
        self.eval()

        predictions = [] 
        for batch in loader: 
            
            Ls = batch["Ls"]
            seq_ids = batch["seq_ids"]
            sequences = batch["sequences"]
            X = batch["seq_embs_pad"].to(self.device)
            with torch.no_grad():
                y_pred = self(X)
            
            for k in range(len(y_pred)):
                predictions.append((
                    seq_ids[k],
                    sequences[k],
                    mat2bp(
                        y_pred[k, : Ls[k], : Ls[k]].squeeze().cpu()
                    )                         
                ))
        predictions = pd.DataFrame(predictions, columns=["id", "sequence", "base_pairs"])

        return predictions
