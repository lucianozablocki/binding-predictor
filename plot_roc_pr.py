import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

TRIAL = "000"  # change to the trial number you want to plot
TRIALS_DIR = "results/trials"  # change to your out_path/trials

fpr = np.load(f"{TRIALS_DIR}/trial_{TRIAL}_roc_fpr.npy")
tpr = np.load(f"{TRIALS_DIR}/trial_{TRIAL}_roc_tpr.npy")
precision = np.load(f"{TRIALS_DIR}/trial_{TRIAL}_pr_precision.npy")
recall = np.load(f"{TRIALS_DIR}/trial_{TRIAL}_pr_recall.npy")

roc_auc = auc(fpr, tpr)
pr_auc = auc(recall, precision)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC curve
ax1.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
ax1.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.legend(loc="lower right")

# PR curve
ax2.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {pr_auc:.3f}")
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"roc_pr_trial_{TRIAL}.png", dpi=150)
plt.show()
