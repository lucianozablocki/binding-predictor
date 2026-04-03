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
        conv_dim=64, kernel_size=16,
        negative_weight=0.1,
        energy_emb_dim=32,
        device='cpu', lr=1e-5
    ):
        super().__init__()
        self.lr = lr
        self.threshold = 0.1
        self.linear_in = nn.Linear(embed_dim, (int) (conv_dim/2))
        # After concat with energy embedding: linear_out + energy_emb_dim per position
        # outer_concat doubles that: 2 * (conv_dim/2 + energy_emb_dim) channels
        outer_dim = 2 * (int(conv_dim/2) + energy_emb_dim)
        self.conv_out = nn.Conv1d(outer_dim, 1, kernel_size=kernel_size, padding="same")
        self.device = device
        self.class_weight = torch.tensor([negative_weight, 1.0]).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.to(device)

    def loss_func(self, yhat, y):
        """yhat and y are [N, M]"""
        # print("yhat shape:", yhat.shape)
        # print("y shape:", y.shape)
        mask = (y != -1)
        loss = binary_cross_entropy_with_logits(yhat[mask], y[mask])
        return loss

    # configurar si usar energy matrix o no, como ablacion
    def forward(self, x, energy_embs):
        B, L, _ = x.shape

        x = self.linear_in(x)  # (B, L, 32)
        x = torch.cat([x, energy_embs], dim=-1)  # (B, L, 32 + 32)
        x = outer_concat(x, x)  # (B, L, L, 128)

        x = x.permute(0, 3, 1, 2)  # (B, 128, L, L)

        x = x.mean(dim=-1)  # (B, 128, L)
        x = self.conv_out(x)  # (B, 1, L)

        return x.squeeze(1)

    def fit(self, loader):
        self.train()
        loss_acum = 0
        f1_acum = 0
        for batch in tqdm(loader):
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)
            energy_embs = batch[4].to(self.device)
            y_pred = self(X, energy_embs)
            # print(f"y_pred size: {y_pred.shape}") # torch.Size([4, 512, 512])
            # print(f"y size: {y.shape}") # torch.Size([4, 512, 512])
            loss = self.loss_func(y_pred, y)
            loss_acum += loss.item()
            f1_acum += binary_f1(y.cpu(), y_pred.detach().cpu())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # self.lr_scheduler.step()
        loss_acum /= len(loader)
        f1_acum /= len(loader)
        return {"loss": loss_acum, "f1": f1_acum}

    def test(self, loader):
        self.eval()
        loss_acum = 0
        f1_acum = 0
        for batch in loader:
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)
            energy_embs = batch[4].to(self.device)
            with torch.no_grad():
                y_pred = self(X, energy_embs)
                loss = self.loss_func(y_pred, y)
            loss_acum += loss.item()

            f1_acum += binary_f1(y.cpu(), y_pred.detach().cpu())
        loss_acum /= len(loader)
        f1_acum /= len(loader)

        return {"loss": loss_acum, "f1": f1_acum}

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
