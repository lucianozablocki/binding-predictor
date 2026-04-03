#!/usr/bin/env python
"""
Pre-compute AIUPred energy embeddings for all sequences.

This script reads all protein FASTA sequences and generates per-residue
energy embeddings using the AIUPred transformer model. Each sequence produces
an (L, 32) embedding array saved as a .npy file.

Usage:
    python precompute_energy_embeddings.py \
        --seq-dir iupred2a/data/seq \
        --output-dir data/energy_embeddings
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from aiupred import AIUPred

# Add src to path for imports
sys.path.insert(0, "src")
from helper_functions import read_fasta


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute AIUPred energy embeddings for all sequences"
    )
    parser.add_argument(
        "--seq-dir",
        type=str,
        default="iupred2a/data/seq",
        help="Directory containing FASTA sequence files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/energy_embeddings",
        help="Output directory for energy embedding .npy files",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize AIUPred predictor (loads model weights once)
    print("Initializing AIUPred predictor...")
    predictor = AIUPred()

    # Get all FASTA files
    seq_dir = Path(args.seq_dir)
    fasta_files = sorted(seq_dir.glob("*.fasta"))
    print(f"Found {len(fasta_files)} FASTA files in {args.seq_dir}")

    for fasta_path in tqdm(fasta_files, desc="Computing energy embeddings"):
        accession = fasta_path.stem

        output_path = Path(args.output_dir) / f"{accession}.npy"
        if output_path.exists():
            continue

        try:
            fasta_dict = read_fasta(str(fasta_path))
            if not fasta_dict:
                print(f"Warning: Empty FASTA for {accession}")
                continue

            sequence = list(fasta_dict.values())[0]

            # Get center-only embedding: shape (L, 32)
            embedding = predictor.get_embedding(sequence, center_only=True)

            np.save(output_path, embedding)

        except Exception as e:
            print(f"Error processing {accession}: {e}")
            continue

    print(f"Done! Energy embeddings saved to {args.output_dir}")


if __name__ == "__main__":
    main()
