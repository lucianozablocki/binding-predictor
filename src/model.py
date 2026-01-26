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

class SecondaryStructurePredictor(nn.Module):
    def __init__(
        self, embed_dim, num_blocks=2,
        conv_dim=64, kernel_size=3,
        negative_weight=0.1,
        device='cpu', lr=1e-5
    ):
        super().__init__()
        self.lr = lr
        self.threshold = 0.1
        self.linear_in = nn.Linear(embed_dim, (int) (conv_dim/2))
        self.resnet = ResNet2D(conv_dim+1, num_blocks, kernel_size)
        self.conv_out = nn.Conv1d(conv_dim+1, 1, kernel_size=kernel_size, padding="same")
        self.device = device
        self.class_weight = torch.tensor([negative_weight, 1.0]).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=2000)

        self.to(device)

    def loss_func(self, yhat, y):
        """yhat and y are [N, M]"""
        # print("yhat shape:", yhat.shape)
        # print("y shape:", y.shape)
        mask = (y != -1)
        loss = binary_cross_entropy_with_logits(yhat[mask], y[mask])
        return loss

    def forward(self, x, accessions):
        # Load pre-computed energy matrices from files
        B, L, _ = x.shape
        energy_matrices = []
        for acc in accessions:
            mat = np.load(f"{ENERGY_MATRICES_DIR}/{acc}.npy")
            energy_matrices.append(torch.tensor(mat, dtype=x.dtype, device=x.device))
        
        # Pad energy matrices to match the max sequence length in batch
        expanded_energy_matrix = torch.zeros((B, L, L), dtype=x.dtype, device=x.device)
        for i, mat in enumerate(energy_matrices):
            seq_len = mat.shape[0]
            expanded_energy_matrix[i, :seq_len, :seq_len] = mat
        x = self.linear_in(x) 
        x = outer_concat(x, x)
        # Add energy matrix as extra channel => (B, L, L, linear_out_dim*2 + 1)
        x = torch.cat((x, expanded_energy_matrix.unsqueeze(-1)), dim=-1)
        x = x.permute(0, 3, 1, 2) 

        x = self.resnet(x)

        x = x.mean(dim=-1) 
        x = self.conv_out(x)


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
        self.lr_scheduler.step()
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
