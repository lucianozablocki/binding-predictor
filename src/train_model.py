import numpy as np
import argparse
import torch
import logging
import sys
import os
import csv
import pandas as pd
import optuna
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

from model import BindingPredictor
from binding_dataset import BindingDataset, pad_collate
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Applied workaround for CuDNN issue, install nvrtc.so
# Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR

parser = argparse.ArgumentParser()

# parser.add_argument("--batch_size", default=4, type=int, help="Batch size to use in forward pass.")
# parser.add_argument("--max_epochs", default=15, type=int, help="Maximum number of training epochs.")
# parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for the training.")
# parser.add_argument("--out_path", default='results', type=str, help="Path to write predictions (base pairs of test partition), weights and logs")
parser.add_argument("--energy_emb_dir", default='data/energy_embeddings', type=str, help="Directory with precomputed energy embeddings (.npy)")
parser.add_argument("--zone_annotations", nargs="+", default=["disorder"],
                    help="Annotation labels to define the positive zone for split confusion metrics.")

parser.add_argument("--max_epochs", default=350, type=int, help="Maximum number of training epochs.")
parser.add_argument("--n_trials", default=50, type=int, help="Number of Optuna trials.")
parser.add_argument("--out_path", default='results', type=str, help="Path to write results and logs.")
batch_size=4
args = parser.parse_args()

# if torch.cuda.is_available():
#     device = f"cuda:{torch.cuda.current_device()}"
# else:
#     device = 'cpu'

device="cuda:0"

os.makedirs(args.out_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(args.out_path, 'log.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# Load datasets once, create loaders per trial (batch_size may vary)
train_dataset = BindingDataset(
    tsv_file='iupred2a/data/2312_disprot/train.tsv',
    seq_dir='iupred2a/data/seq',
    energy_emb_dir=args.energy_emb_dir,
)

val_dataset = BindingDataset(
    tsv_file='iupred2a/data/2312_disprot/val.tsv',
    seq_dir='iupred2a/data/seq',
    energy_emb_dir=args.energy_emb_dir,
)

embed_dim = 1280  # ESM2 representations
pos_weight = 3.54

def objective(trial):
    # Model hyperparameters
    linear_dim = trial.suggest_categorical("linear_dim", [32])
    num_blocks = trial.suggest_categorical("num_blocks", [2])
    kernel_size = trial.suggest_categorical("kernel_size", [9])
    dropout = trial.suggest_categorical("dropout", [0.3])

    # Training hyperparameters
    lr = trial.suggest_float("lr", 6.61e-04, 6.61e-04, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW"])
    weight_decay = trial.suggest_categorical("weight_decay", [0.1])
    # batch_size = trial.suggest_categorical("batch_size", [4])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    net = BindingPredictor(
        embed_dim=embed_dim,
        linear_dim=linear_dim,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        dropout=dropout,
        pos_weight=pos_weight,
        device=device,
    )

    num_params = sum(p.numel() for p in net.parameters())
    logger.info(f"Trial {trial.number}: linear_dim={linear_dim}, num_blocks={num_blocks}, "
                f"kernel_size={kernel_size}, dropout={dropout}, lr={lr:.2e}, "
                f"optimizer={optimizer_name}, weight_decay={weight_decay}, "
                f"pos_weight={pos_weight:.4f}, params={num_params}, batch_size={batch_size}")

    # Per-trial metrics CSV with parameters as header comments
    trial_dir = os.path.join(args.out_path, "trials")
    os.makedirs(trial_dir, exist_ok=True)
    metrics_path = os.path.join(trial_dir, f"trial_{trial.number:03d}.csv")
    with open(metrics_path, 'w', newline='') as f:
        f.write(f"# trial={trial.number} linear_dim={linear_dim} num_blocks={num_blocks} "
                f"kernel_size={kernel_size} dropout={dropout} lr={lr:.2e} "
                f"optimizer={optimizer_name} weight_decay={weight_decay} "
                f"pos_weight={pos_weight:.4f} params={num_params} batch_size={batch_size} "
                f"zone_annotations={args.zone_annotations}\n")
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_f1", "train_tn", "train_fp", "train_fn", "train_tp",
            "train_zone_tn", "train_zone_fp", "train_zone_fn", "train_zone_tp",
            "train_non_zone_tn", "train_non_zone_fp", "train_non_zone_fn", "train_non_zone_tp",
            "val_loss", "val_f1", "val_tn", "val_fp", "val_fn", "val_tp",
            "val_zone_tn", "val_zone_fp", "val_zone_fn", "val_zone_tp",
            "val_non_zone_tn", "val_non_zone_fp", "val_non_zone_fn", "val_non_zone_tp",
        ])

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    for epoch in range(args.max_epochs):
        train_metrics = net.fit(train_loader, optimizer)
        val_metrics = net.test(val_loader)

        logger.info(f"  Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f} "
                     f"train_f1={train_metrics['f1']:.4f} "
                     f"val_loss={val_metrics['loss']:.4f} "
                     f"val_f1={val_metrics['f1']:.4f} "
                     f"val_zone_tp={val_metrics['zone_tp']} "
                     f"val_non_zone_tp={val_metrics['non_zone_tp']}")

        # Append to per-trial CSV (flush each row so data survives crashes)
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_metrics['loss']:.4f}", f"{train_metrics['f1']:.4f}",
                train_metrics['tn'], train_metrics['fp'], train_metrics['fn'], train_metrics['tp'],
                train_metrics['zone_tn'], train_metrics['zone_fp'], train_metrics['zone_fn'], train_metrics['zone_tp'],
                train_metrics['non_zone_tn'], train_metrics['non_zone_fp'], train_metrics['non_zone_fn'], train_metrics['non_zone_tp'],
                f"{val_metrics['loss']:.4f}", f"{val_metrics['f1']:.4f}",
                val_metrics['tn'], val_metrics['fp'], val_metrics['fn'], val_metrics['tp'],
                val_metrics['zone_tn'], val_metrics['zone_fp'], val_metrics['zone_fn'], val_metrics['zone_tp'],
                val_metrics['non_zone_tn'], val_metrics['non_zone_fp'], val_metrics['non_zone_fn'], val_metrics['non_zone_tp'],
            ])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            weights_path = os.path.join(trial_dir, f"trial_{trial.number:03d}_best_weights.pmt")
            torch.save(net.state_dict(), weights_path)
            logger.info(f"  Saved best weights (val_f1={best_val_f1:.4f}) -> {weights_path}")

        # Report intermediate value for pruning
        trial.report(val_metrics["f1"], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    y_true, y_prob = net.collect_scores(val_loader)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    np.save(os.path.join(trial_dir, f"trial_{trial.number:03d}_roc_fpr.npy"), fpr)
    np.save(os.path.join(trial_dir, f"trial_{trial.number:03d}_roc_tpr.npy"), tpr)
    logger.info(f"  Trial {trial.number} ROC AUC: {auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    np.save(os.path.join(trial_dir, f"trial_{trial.number:03d}_pr_precision.npy"), precision)
    np.save(os.path.join(trial_dir, f"trial_{trial.number:03d}_pr_recall.npy"), recall)
    logger.info(f"  Trial {trial.number} Average Precision: {ap:.4f}")

    return best_val_f1


study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    study_name="binding_predictor_hparam_search",
)

logger.info(f"Starting Optuna study with {args.n_trials} trials on {device}")
study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

# Log and save results
logger.info(f"Best trial: {study.best_trial.number}")
logger.info(f"  Best val F1: {study.best_trial.value:.4f}")
logger.info(f"  Best params: {study.best_trial.params}")

trials_df = study.trials_dataframe()
trials_df.to_csv(os.path.join(args.out_path, "optuna_trials.csv"), index=False)
logger.info(f"All trial results saved to {os.path.join(args.out_path, 'optuna_trials.csv')}")
