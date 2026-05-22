"""
Positive class analysis for the binding prediction dataset.

Reuses BindingDataset to get the exact same data the model sees (same filters,
same region merging), then computes:
  1. Residue-level class balance (total 0s vs 1s)
  2. Binding region size distribution (absolute)
  3. Binding region size relative to sequence length
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np

from binding_dataset import BindingDataset

# TSV_FILE = os.path.join(os.path.dirname(__file__), '..', 'iupred2a', 'data', 'disprot_v_25_06.tsv')
TSV_FILE = os.path.join(os.path.dirname(__file__), '..', 'iupred2a', 'data', 'DisProt_2023_12_IDPO-GO.tsv')
SEQ_DIR = os.path.join(os.path.dirname(__file__), '..', 'iupred2a', 'data', 'seq')
OUT_DIR = os.path.join(os.path.dirname(__file__), '2312_disprot')


def main():
    # Load the dataset (same filtering/merging as training)
    dataset = BindingDataset(tsv_file=TSV_FILE, seq_dir=SEQ_DIR)
    print(f"Loaded {len(dataset)} proteins")

    # --- 1. Residue-level class balance ---
    total_ones = 0
    total_zeros = 0

    # Per-protein stats for region analysis
    region_lengths = []        # absolute binding region sizes
    relative_coverages = []    # binding_residues / seq_len per protein
    seq_lengths = []

    for encoded_seq, target_mask, zone_mask, accession, energy_embs in dataset:
        seq_len = target_mask.shape[0]
        ones = int(target_mask.sum().item())
        zeros = seq_len - ones

        total_ones += ones
        total_zeros += zeros
        seq_lengths.append(seq_len)
        relative_coverages.append(ones / seq_len)

        # Extract contiguous binding regions from the merged mask
        # to get individual region sizes
        in_region = False
        region_start = 0
        for i in range(seq_len):
            if target_mask[i] == 1.0 and not in_region:
                in_region = True
                region_start = i
            elif target_mask[i] == 0.0 and in_region:
                in_region = False
                region_lengths.append(i - region_start)
            # also collect region relative to seq length
        if in_region:
            region_lengths.append(seq_len - region_start)

    total = total_ones + total_zeros
    print(f"\n=== Residue-level class balance ===")
    print(f"  Total residues:  {total:,}")
    print(f"  Binding (1):     {total_ones:,}  ({100*total_ones/total:.2f}%)")
    print(f"  Non-binding (0): {total_zeros:,}  ({100*total_zeros/total:.2f}%)")
    print(f"  Ratio 0/1:       {total_zeros/total_ones:.1f}:1")

    print(f"\n=== Binding regions ===")
    print(f"  Total contiguous regions: {len(region_lengths)}")
    print(f"  Mean region length:  {np.mean(region_lengths):.1f}")
    print(f"  Median region length: {np.median(region_lengths):.1f}")
    print(f"  Min: {np.min(region_lengths)}, Max: {np.max(region_lengths)}")

    # --- Plot 1: Residue-level balance bar chart ---
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Non-binding (0)', 'Binding (1)'], [total_zeros, total_ones],
                  color=['#4C72B0', '#DD8452'])
    ax.set_ylabel('Residue count')
    ax.set_title('Residue-level class balance')
    for bar, val in zip(bars, [total_zeros, total_ones]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:,}\n({100*val/total:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'residue_class_balance.png'), dpi=150)
    print(f"\nSaved residue_class_balance.png")

    # --- Plot 2: Binding region length histogram ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(region_lengths, bins=50, color='#DD8452', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Binding region length (residues)')
    ax.set_ylabel('Count')
    ax.set_title(f'Binding region size distribution (n={len(region_lengths)})')
    ax.axvline(np.median(region_lengths), color='red', linestyle='--', label=f'median={np.median(region_lengths):.0f}')
    ax.axvline(np.mean(region_lengths), color='blue', linestyle='--', label=f'mean={np.mean(region_lengths):.0f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'binding_region_lengths.png'), dpi=150)
    print(f"Saved binding_region_lengths.png")

    # --- Plot 3: Relative binding coverage per protein ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(relative_coverages, bins=50, color='#55A868', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Binding coverage (binding residues / sequence length)')
    ax.set_ylabel('Count (proteins)')
    ax.set_title(f'Per-protein binding coverage (n={len(relative_coverages)})')
    ax.axvline(np.median(relative_coverages), color='red', linestyle='--',
               label=f'median={np.median(relative_coverages):.2f}')
    ax.axvline(np.mean(relative_coverages), color='blue', linestyle='--',
               label=f'mean={np.mean(relative_coverages):.2f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'binding_coverage_per_protein.png'), dpi=150)
    print(f"Saved binding_coverage_per_protein.png")

    # --- Plot 4: Binding region length relative to its protein's length ---
    # For each region, compute region_len / seq_len of the protein it belongs to
    relative_region_lengths = []
    for encoded_seq, target_mask, zone_mask, accession, energy_embs in dataset:
        seq_len = target_mask.shape[0]
        in_region = False
        region_start = 0
        for i in range(seq_len):
            if target_mask[i] == 1.0 and not in_region:
                in_region = True
                region_start = i
            elif target_mask[i] == 0.0 and in_region:
                in_region = False
                relative_region_lengths.append((i - region_start) / seq_len)
        if in_region:
            relative_region_lengths.append((seq_len - region_start) / seq_len)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(relative_region_lengths, bins=50, color='#8172B2', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Region length / Sequence length')
    ax.set_ylabel('Count (regions)')
    ax.set_title(f'Relative binding region size (n={len(relative_region_lengths)})')
    ax.axvline(np.median(relative_region_lengths), color='red', linestyle='--',
               label=f'median={np.median(relative_region_lengths):.2f}')
    ax.axvline(np.mean(relative_region_lengths), color='blue', linestyle='--',
               label=f'mean={np.mean(relative_region_lengths):.2f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'binding_region_relative_lengths.png'), dpi=150)
    print(f"Saved binding_region_relative_lengths.png")

    plt.close('all')


if __name__ == '__main__':
    main()
