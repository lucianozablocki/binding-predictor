import numpy as np
import argparse
import torch
import pandas as pd
import logging
import os
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from model import BindingPredictor
from binding_dataset import CSVBindingDataset, pad_collate
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Applied workaround for CuDNN issue, install nvrtc.so
# Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR

parser = argparse.ArgumentParser()
parser.add_argument("--test_csv", type=str, required=True,
                    help="CSV file with accession, sequence, target columns.")
parser.add_argument("--esm2_repr_path", type=str, required=True,
                    help="Path to .pt file with precomputed ESM2 representations.")
parser.add_argument("--energy_emb_dir", default='data/energy_embeddings', type=str,
                    help="Directory with precomputed energy embeddings (.npy)")
parser.add_argument("--weights_path", type=str, required=True,
                    help="Path to saved model weights (.pmt/.pt).")
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--out_path", default='results', type=str,
                    help="Directory to write metrics and predictions.")

# Model architecture (must match saved weights)
parser.add_argument("--linear_dim", default=32, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--kernel_size", default=9, type=int)
parser.add_argument("--dropout", default=0.2, type=float)

args = parser.parse_args()

if torch.cuda.is_available():
    device = f"cuda:{torch.cuda.current_device()}"
else:
    device = 'cpu'

os.makedirs(args.out_path, exist_ok=True)

out_name = os.path.splitext(os.path.basename(args.out_path))[0]

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(args.out_path, f'log-{out_name}.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Device: {device}")
logger.info(f"Test CSV: {args.test_csv}")
logger.info(f"Weights: {args.weights_path}")

test_dataset = CSVBindingDataset(
    csv_file=args.test_csv,
    esm2_repr_path=args.esm2_repr_path,
    energy_emb_dir=args.energy_emb_dir,
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                         shuffle=False, collate_fn=pad_collate)

embed_dim = 1280  # ESM2
pos_weight = 3.54
linear_dim = args.linear_dim
num_blocks = args.num_blocks
kernel_size = args.kernel_size
dropout = args.dropout

net = BindingPredictor(
    embed_dim=embed_dim,
    linear_dim=linear_dim,
    num_blocks=num_blocks,
    kernel_size=kernel_size,
    dropout=dropout,
    pos_weight=pos_weight,
    device=device,
)
net.load_state_dict(torch.load(args.weights_path, map_location=device))
net.eval()
logger.info("Model loaded.")

metrics = net.test(test_loader)
metrics = {f"test_{k}": v for k, v in metrics.items()}
logger.info(" ".join([f"{k}: {v}" for k, v in metrics.items()]))

y_true, y_prob = net.collect_scores(test_loader)
metrics["test_roc_auc"] = roc_auc_score(y_true, y_prob)
metrics["test_avg_precision"] = average_precision_score(y_true, y_prob)
logger.info(f"ROC AUC: {metrics['test_roc_auc']:.4f}  Avg Precision: {metrics['test_avg_precision']:.4f}")

fpr, tpr, _ = roc_curve(y_true, y_prob)
np.save(os.path.join(args.out_path, f"{out_name}_roc_fpr.npy"), fpr)
np.save(os.path.join(args.out_path, f"{out_name}_roc_tpr.npy"), tpr)
logger.info(f"ROC curve saved to {args.out_path}/{out_name}_roc_{{fpr,tpr}}.npy")

precision, recall, _ = precision_recall_curve(y_true, y_prob)
np.save(os.path.join(args.out_path, f"{out_name}_pr_precision.npy"), precision)
np.save(os.path.join(args.out_path, f"{out_name}_pr_recall.npy"), recall)
logger.info(f"PR curve saved to {args.out_path}/{out_name}_pr_{{precision,recall}}.npy")

metrics_path = os.path.join(args.out_path, f"{out_name}.csv")
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
logger.info(f"Metrics saved to {metrics_path}")

# Per-residue predictions CSV
rows = []
net.eval()
with torch.no_grad():
    for batch in test_loader:
        X = batch[0].to(device)
        y = batch[1]
        energy_embs = batch[5].to(device)
        accessions = batch[4]
        lengths = batch[3]
        y_pred = torch.sigmoid(net(X, energy_embs)).cpu()
        for i, acc in enumerate(accessions):
            L = lengths[i].item()
            for pos in range(L):
                rows.append({
                    "accession": acc,
                    "position": pos + 1,
                    "target": int(y[i, pos].item()),
                    "score": float(y_pred[i, pos].item()),
                })

preds_path = os.path.join(args.out_path, f"{out_name}_preds.csv")
pd.DataFrame(rows).to_csv(preds_path, index=False)
logger.info(f"Predictions saved to {preds_path}")
