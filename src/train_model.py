import numpy as np
import argparse
import torch
import logging
import os
import pandas as pd

from model import BindingPredictor
# from dataset import create_dataloader
from binding_dataset import get_binding_dataloader
from utils import get_embed_dim

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Applied workaround for CuDNN issue, install nvrtc.so
# Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR

parser = argparse.ArgumentParser()

# parser.add_argument("--train_partition_path", type=str, help="The path of the train partition.")
# parser.add_argument("--val_partition_path", type=str, help="The path of the validation partition.")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size to use in forward pass.")
parser.add_argument("--max_epochs", default=15, type=int, help="Maximum number of training epochs.")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for the training.")
parser.add_argument("--out_path", default='results', type=str, help="Path to write predictions (base pairs of test partition), weights and logs")
parser.add_argument("--energy_emb_dir", default='data/energy_embeddings', type=str, help="Directory with precomputed energy embeddings (.npy)")

args = parser.parse_args()

if torch.cuda.is_available():
    device=f"cuda:{torch.cuda.current_device()}"
else:
    device='cpu'
# device='cuda:1'
os.makedirs(args.out_path, exist_ok=True)

# embeddings_path = f"data/embeddings/{args.emb}.h5"

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(args.out_path, f'log.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

logger.info("using lr: {}".format(args.lr))

train_loader = get_binding_dataloader(
    tsv_file='iupred2a/data/train.tsv',
    seq_dir='iupred2a/data/seq',
    energy_emb_dir=args.energy_emb_dir,
    batch_size=args.batch_size
)

val_loader = get_binding_dataloader(
    tsv_file='iupred2a/data/val.tsv',
    seq_dir='iupred2a/data/seq',
    energy_emb_dir=args.energy_emb_dir,
    batch_size=args.batch_size,
)

# if args.val_partition_path:
#     val_loader = create_dataloader(
#         embeddings_path,
#         args.val_partition_path,
#         args.batch_size,
#         False
#     )

embed_dim = get_embed_dim(train_loader)
net = BindingPredictor(embed_dim=embed_dim, device=device, lr=args.lr)
# print model amount of parameters
num_params = sum(p.numel() for p in net.parameters())
logger.info(f"Model initialized with {num_params} parameters")
metrics_for_epoch = []
logger.info(f"Run on {args.out_path}, with device {device}")
logger.info(f"Training with file: {train_loader}, batch size: {args.batch_size}")
# if args.val_partition_path:
#     logger.info(f"Validation enabled, using file: {args.val_partition_path}")

for epoch in range(args.max_epochs):
    logger.info(f"Starting epoch {epoch+1}")
    metrics = net.fit(train_loader)
    
    metrics = {f"train_{k}": v for k, v in metrics.items()}

    # if args.val_partition_path:
    logger.info("Running validation inference")
    val_metrics = net.test(val_loader)
       
    val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
    metrics.update(val_metrics)

    metrics_for_epoch.append(metrics)
    logger.info(" ".join([f"{k}: {v:.3f}" for k, v in metrics.items()]))    

pd.set_option('display.float_format','{:.3f}'.format)
pd.DataFrame(metrics_for_epoch).to_csv(os.path.join(args.out_path, f"metrics.csv"), index=False)

# torch.save(
#     net.state_dict(),
#     os.path.join(args.out_path, f"weights.pmt")
# )
