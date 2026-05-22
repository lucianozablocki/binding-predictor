"""Plot the score distribution of positives vs. negatives from a per-residue
predictions CSV produced by `src/test_model.py`.

The CSV is expected to have columns: accession, position, target, score
(with `target` in {0, 1} and `score` in [0, 1]).

Why this plot:
- ROC AUC measures only the ranking of positive vs. negative scores. Two models
  with the same AUC can have very different score distributions, which affects
  calibration and how easy further improvements are.
- Looking at the overlap of the two distributions is the most direct diagnostic
  for "where is the AUC bottleneck?" — heavy overlap = limited separability,
  bimodal-ish = model has learned something cleanly separable.

Usage:
    python plot_score_dist.py --preds_csv path/to/<name>_preds.csv \\
        [--out_path path/to/<name>_score_dist.png] \\
        [--title "Optional title"] \\
        [--bins 50] [--no_log_y]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def plot_score_distribution(
    preds_csv: str,
    out_path: str,
    title: str | None,
    bins: int,
    log_y: bool,
) -> None:
    df = pd.read_csv(preds_csv)
    required = {"target", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{preds_csv} is missing columns: {missing}")

    pos = df.loc[df["target"] == 1, "score"].values
    neg = df.loc[df["target"] == 0, "score"].values

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(
            f"Need both classes in {preds_csv}: got pos={len(pos)}, neg={len(neg)}"
        )

    auc = roc_auc_score(df["target"].values, df["score"].values)
    aps = average_precision_score(df["target"].values, df["score"].values)
    prevalence = len(pos) / (len(pos) + len(neg))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ax.hist(neg, bins=edges, color="#4C72B0", alpha=0.55, label=f"Negative (n={len(neg):,})",
            edgecolor="white", linewidth=0.3)
    ax.hist(pos, bins=edges, color="#DD8452", alpha=0.65, label=f"Positive (n={len(pos):,})",
            edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Predicted score (sigmoid output)")
    ax.set_ylabel("Residue count" + (" (log)" if log_y else ""))
    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)

    base_title = title or os.path.splitext(os.path.basename(preds_csv))[0]
    ax.set_title(
        f"{base_title}\n"
        f"AUC={auc:.4f}  APS={aps:.4f}  prevalence={prevalence:.3f}"
    )
    ax.legend(loc="upper center")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved score distribution -> {out_path}")
    print(f"  AUC={auc:.4f}  APS={aps:.4f}  prevalence={prevalence:.3f}")
    print(f"  positives: n={len(pos):,}  mean={pos.mean():.3f}  median={np.median(pos):.3f}")
    print(f"  negatives: n={len(neg):,}  mean={neg.mean():.3f}  median={np.median(neg):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--preds_csv", required=True,
        help="Path to <out_name>_preds.csv produced by src/test_model.py.",
    )
    parser.add_argument(
        "--out_path", default=None,
        help="Where to save the PNG. Defaults to <preds_csv stem>_score_dist.png "
             "in the same directory as --preds_csv.",
    )
    parser.add_argument(
        "--title", default=None,
        help="Optional plot title. Defaults to the preds_csv basename.",
    )
    parser.add_argument("--bins", default=50, type=int, help="Number of histogram bins.")
    parser.add_argument(
        "--no_log_y", action="store_true",
        help="Disable log-scale y-axis (default is log to make positives visible).",
    )
    args = parser.parse_args()

    out_path = args.out_path
    if out_path is None:
        stem = os.path.splitext(args.preds_csv)[0]
        # If the filename already ends in `_preds`, strip that for a cleaner output name.
        if stem.endswith("_preds"):
            stem = stem[: -len("_preds")]
        out_path = f"{stem}_score_dist.png"

    plot_score_distribution(
        preds_csv=args.preds_csv,
        out_path=out_path,
        title=args.title,
        bins=args.bins,
        log_y=not args.no_log_y,
    )


if __name__ == "__main__":
    main()
