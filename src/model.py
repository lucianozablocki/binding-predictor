import torch.nn as nn
import torch
import numpy as np
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import LinearLR
import pandas as pd
from metrics import binary_f1
from utils import mat2bp, outer_concat
from tqdm import tqdm

ENERGY_MATRICES_DIR = "data/expanded_energy_matrices"
DEFAULT_ENERGY_MATRIX_PATH = "iupred2a/data/iupred2_long_energy_matrix"

# Amino acid vocabulary (same order as binding_dataset.py)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def read_energy_matrix(filepath: str) -> np.ndarray:
    """Read the 20x20 amino acid energy matrix from file."""
    n = len(AMINO_ACIDS)
    matrix = np.zeros((n, n), dtype=np.float32)
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            aa1, aa2, value = parts[0], parts[1], float(parts[2])
            if aa1 in AA_TO_INDEX and aa2 in AA_TO_INDEX:
                i = AA_TO_INDEX[aa1]
                j = AA_TO_INDEX[aa2]
                matrix[i, j] = value
    return matrix

class ResNet2DBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=False):
        super().__init__()

        # Bottleneck architecture
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, bias=bias, padding="same"),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = x

        x = self.conv_net(x)
        x = x + residual

        return x

class ResNet2D(nn.Module):
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNet2DBlock(embed_dim, kernel_size, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class BindingPredictor(nn.Module):
    def __init__(
        self, embed_dim, num_blocks=1,
        linear_dim=32, kernel_size=16,
        reduce_op='mean',
        negative_weight=0.1,
        energy_emb_dim=32,
        device='cpu', lr=1e-5,
        energy_matrix_path=DEFAULT_ENERGY_MATRIX_PATH,
    ):
        super().__init__()
        # conv_dim = linear_dim * 2  # outer_concat doubles the dim
        self.reduce_op = reduce_op
        self.threshold = 0.1
        # self.linear_in = nn.Linear(embed_dim, (int) (conv_dim/2))
        # After concat with energy embedding: linear_out + energy_emb_dim per position
        # outer_concat doubles that: 2 * (conv_dim/2 + energy_emb_dim) channels
        outer_dim = 2 * (linear_dim + energy_emb_dim)
        # self.conv_out = nn.Conv1d(outer_dim, 1, kernel_size=kernel_size, padding="same")
        # self.device = device
        # self.class_weight = torch.tensor([negative_weight, 1.0]).float().to(self.device)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.linear_in = nn.Linear(embed_dim, linear_dim)
        # Learnable energy expansion: fixed 20x20 matrix scaled by learnable weights
        # energy_np = read_energy_matrix(energy_matrix_path)
        # self.register_buffer('energy_matrix', torch.tensor(energy_np))  # (20, 20) fixed
        # self.energy_weights = nn.Parameter(torch.ones(20, 20))          # (20, 20) learnable
        # self.energy_proj = nn.Conv2d(in_channels=1, out_channels=conv_dim, kernel_size=1, bias=False)
        self.conv_out = nn.Conv1d(outer_dim, 1, kernel_size=kernel_size, padding="same")
        self.device = device
        self.class_weight = torch.tensor([negative_weight, 1.0]).float().to(self.device)

        self.to(device)

    def loss_func(self, yhat, y):
        """yhat and y are [N, M]"""
        # print("yhat shape:", yhat.shape)
        # print("y shape:", y.shape)
        mask = (y != -1)
        loss = binary_cross_entropy_with_logits(yhat[mask], y[mask])
        return loss

    _energy_cache = {}

    # configurar si usar energy matrix o no, como ablacion
    def forward(self, x, energy_embs):
        B, L, _ = x.shape

        x = self.linear_in(x)  # (B, L, 64)
        x = torch.cat([x, energy_embs], dim=-1)  # (B, L, 64 + 32)
        x = outer_concat(x, x)  # (B, L, L, 128)

        x = x.permute(0, 3, 1, 2)  # (B, 128, L, L)

        x = x.mean(dim=-1)  # (B, 128, L)
        x = self.conv_out(x)  # (B, 1, L)

        #######

        # x = x.permute(0, 3, 1, 2)  # (B, conv_dim, L, L)

        # # Project energy matrix from 1 channel to conv_dim channels and add
        # energy = expanded_energy_matrix.unsqueeze(1)  # (B, 1, L, L)
        # energy = self.energy_proj(energy)              # (B, conv_dim, L, L)
        # x = x + energy

        # # x = self.resnet(x)
        # # B X 65 x L x L
        # if self.reduce_op == 'mean':
        #     x = x.mean(dim=-1)
        # elif self.reduce_op == 'max':
        #     x = x.max(dim=-1).values
        # elif self.reduce_op == 'std':
        #     x = x.std(dim=-1)
        # # B x 65 x L x 1
        # # x = self.dropout(x)
        # x = self.conv_out(x)
        # # B x 1 x L

        return x.squeeze(1)

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
            Y = batch[1].to(self.device)
            zone_mask = batch[2].to(self.device)
            lens += (Y != -1).sum().item()
            y = batch[1].to(self.device)
            accessions = batch[4]  # 5th element contains accession IDs (not used as we are loading the energy embeddings at dataset construction time)
            energy_embs = batch[5].to(self.device)
            y_pred = self(X, energy_embs)
            # print(f"y_pred size: {y_pred.shape}") # torch.Size([4, 512, 512])
            # print(f"y size: {y.shape}") # torch.Size([4, 512, 512])
            # y_pred = self(X, accessions)
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
            energy_embs = batch[5].to(self.device)
            zone_mask = batch[2].to(self.device)
            # accessions = batch[4]  # 5th element contains accession IDs
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
