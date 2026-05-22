from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _select_epoch_row(df: pd.DataFrame, epoch="last") -> pd.Series:
	"""Return one row from the trial dataframe based on epoch selector.

	Args:
		df: DataFrame loaded from the trial CSV.
		epoch: "last", an epoch number from the `epoch` column, or an integer
			row index.
	"""
	if df.empty:
		raise ValueError("The CSV has no data rows.")

	if epoch == "last":
		return df.iloc[-1]

	if isinstance(epoch, int):
		if "epoch" in df.columns and (df["epoch"] == epoch).any():
			return df.loc[df["epoch"] == epoch].iloc[-1]

		if epoch < 0 or epoch >= len(df):
			raise IndexError(
				f"Row index {epoch} is out of range for dataframe length {len(df)}."
			)
		return df.iloc[epoch]

	raise ValueError("`epoch` must be 'last' or an integer.")


def _counts_to_matrix(row: pd.Series, prefix: str) -> np.ndarray:
	"""Build [[TN, FP], [FN, TP]] matrix for a given metric prefix."""
	required = [f"{prefix}_tn", f"{prefix}_fp", f"{prefix}_fn", f"{prefix}_tp"]
	missing = [col for col in required if col not in row.index]
	if missing:
		raise KeyError(f"Missing required columns for '{prefix}': {missing}")

	tn = int(row[f"{prefix}_tn"])
	fp = int(row[f"{prefix}_fp"])
	fn = int(row[f"{prefix}_fn"])
	tp = int(row[f"{prefix}_tp"])
	return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def _draw_matrix(ax, matrix: np.ndarray, title: str, cmap: str = "Blues") -> None:
	"""Render one confusion matrix on the provided axis."""
	im = ax.imshow(matrix, cmap=cmap)
	ax.set_title(title)
	ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
	ax.set_yticks([0, 1], labels=["True 0", "True 1"])

	threshold = matrix.max() / 2 if matrix.size else 0
	for i in range(2):
		for j in range(2):
			value = int(matrix[i, j])
			color = "white" if value > threshold else "black"
			ax.text(j, i, f"{value:,}", ha="center", va="center", color=color)

	ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_confussion_matrices_from_trial_csv(
	csv_path,
	epoch="last",
	figsize=(12, 10),
	cmap="Blues",
):
	"""Plot 4 confusion matrices from a trial CSV (train/val x zone/non-zone).

	Args:
		csv_path: Path to a CSV like `trial_000.csv`.
		epoch: "last" (default), epoch number, or row index.
		figsize: Matplotlib figure size.
		cmap: Matplotlib colormap.

	Returns:
		(fig, axes, matrices): Matplotlib objects and raw matrix values.
	"""
	csv_path = Path(csv_path)
	df = pd.read_csv(csv_path, comment="#")
	row = _select_epoch_row(df, epoch=epoch)

	matrices = {
		"Train Zone": _counts_to_matrix(row, "train_zone"),
		"Train Non-Zone": _counts_to_matrix(row, "train_non_zone"),
		"Val Zone": _counts_to_matrix(row, "val_zone"),
		"Val Non-Zone": _counts_to_matrix(row, "val_non_zone"),
	}

	fig, axes = plt.subplots(2, 2, figsize=figsize)
	axes = axes.ravel()

	for ax, (title, matrix) in zip(axes, matrices.items()):
		_draw_matrix(ax, matrix, title=title, cmap=cmap)

	selected_epoch = int(row["epoch"]) if "epoch" in row.index else "N/A"
	fig.suptitle(f"Confusion Matrices - {csv_path.name} - epoch={selected_epoch}")
	fig.tight_layout()

	return fig, axes, matrices

