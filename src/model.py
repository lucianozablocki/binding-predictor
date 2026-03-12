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
        self, embed_dim, num_blocks=2,
        conv_dim=64, kernel_size=3,
        negative_weight=0.1,
        dropout=0.25,
        energy_matrix_path=DEFAULT_ENERGY_MATRIX_PATH,
        device='cpu', lr=1e-5
    ):
        super().__init__()
        self.lr = lr
        self.threshold = 0.1
        self.linear_in = nn.Linear(embed_dim, (int) (conv_dim/2))
        # Learnable energy expansion: fixed 20x20 matrix scaled by learnable weights
        energy_np = read_energy_matrix(energy_matrix_path)
        self.register_buffer('energy_matrix', torch.tensor(energy_np))  # (20, 20) fixed
        self.energy_weights = nn.Parameter(torch.ones(20, 20))          # (20, 20) learnable
        self.energy_proj = nn.Conv2d(in_channels=1, out_channels=conv_dim, kernel_size=1, bias=False)
        # Old version: resnet and conv_out with conv_dim+1 channels (energy matrix concatenated)
        # self.resnet = ResNet2D(conv_dim+1, num_blocks, kernel_size)
        # self.conv_out = nn.Conv1d(conv_dim+1, 1, kernel_size=kernel_size, padding="same")
        self.resnet = ResNet2D(conv_dim, num_blocks, kernel_size)
        # Old version: single conv1d output
        # self.conv_out = nn.Conv1d(conv_dim, 1, kernel_size=kernel_size, padding="same")
        # 2 conv1D out aca
        self.dropout = nn.Dropout1d(p=dropout)
        self.conv_out = nn.Sequential(
            nn.Conv1d(conv_dim, conv_dim // 2, kernel_size=kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout),
            nn.Conv1d(conv_dim // 2, conv_dim // 4, kernel_size=kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout),
            nn.Conv1d(conv_dim // 4, 1, kernel_size=kernel_size, padding="same"),
        )
        self.device = device
        self.class_weight = torch.tensor([negative_weight, 1.0]).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # self.lr_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=2000)

        self.to(device)

    def loss_func(self, yhat, y):
        """yhat and y are [N, M]"""
        # print("yhat shape:", yhat.shape)
        # print("y shape:", y.shape)
        mask = (y != -1)
        loss = binary_cross_entropy_with_logits(yhat[mask], y[mask])
        return loss

    # configurar si usar energy matrix o no, como ablacion
    def forward(self, x, accessions):
        B, L, _ = x.shape

        # Learnable energy expansion: W_ij * E_ij for each AA pair
        # Recover AA indices from one-hot input
        aa_indices = x.argmax(dim=-1)  # (B, L)
        scaled_energy = self.energy_weights * self.energy_matrix  # (20, 20)
        # Expand to (B, L, L) using AA indices
        expanded_energy_matrix = scaled_energy[aa_indices.unsqueeze(2), aa_indices.unsqueeze(1)]  # (B, L, L)

        x = self.linear_in(x) 
        x = outer_concat(x, x)
        
        # Old version: concatenate energy matrix as extra channel
        # como darle mas importancia a la energy matrix aca?
        # sumar/mutiplicar la matriz a todos los canales? 
        # usar conv2d de 1x1 q pase 1 canal a 64, y se sume a todos los canales de la outer concat
        # Add energy matrix as extra channel => (B, L, L, linear_out_dim*2 + 1)
        # x = torch.cat((x, expanded_energy_matrix.unsqueeze(-1)), dim=-1)
        # x = x.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)  # (B, conv_dim, L, L)

        # Project energy matrix from 1 channel to conv_dim channels and add
        energy = expanded_energy_matrix.unsqueeze(1)  # (B, 1, L, L)
        energy = self.energy_proj(energy)              # (B, conv_dim, L, L)
        x = x + energy 

        x = self.resnet(x)
        # B X 65 x L x L
        x = x.mean(dim=-1) # std/max attn->L variable
        # B x 65 x L x 1
        # x = self.dropout(x)
        x = self.conv_out(x)
        # B x 1 x L

        return x.squeeze(1)

    def fit(self, loader):
        self.train()
        loss_acum = 0
        f1_acum = 0
        for batch in tqdm(loader):
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)
            accessions = batch[3]  # 4th element contains accession IDs
            y_pred = self(X, accessions)
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
            accessions = batch[3]  # 4th element contains accession IDs
            with torch.no_grad():
                y_pred = self(X, accessions)
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
