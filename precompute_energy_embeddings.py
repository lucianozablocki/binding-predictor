#!/usr/bin/env python
"""
Pre-compute AIUPred energy embeddings for all sequences.

This script reads all protein FASTA sequences and generates per-residue
energy embeddings using the AIUPred transformer model. Each sequence produces
an (L, 32) embedding array saved as a .npy file.

Usage (per-protein fasta files):
    python precompute_energy_embeddings.py \
        --seq-dir iupred2a/data/seq \
        --output-dir data/energy_embeddings

Usage (combined fasta with format >ACC / SEQUENCE / PREDICTION per record):
    python precompute_energy_embeddings.py \
        --fasta iupred2a/data/caid3/binding.fasta \
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


def read_combined_fasta(path: str) -> list[tuple[str, str]]:
    """Parse a combined fasta where each record is exactly 3 lines:
        >ACCESSION
        SEQUENCE
        PREDICTION  (binary string — ignored here, kept for future use)
    Returns a list of (accession, sequence) pairs.
    """
    records = []
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        if not header:
            i += 1
            continue
        if not header.startswith(">"):
            raise ValueError(f"Expected header line starting with '>'; got: {header!r}")
        acc = header[1:].strip()
        if i + 1 >= len(lines):
            raise ValueError(f"Missing sequence line after header for {acc!r}")
        seq = lines[i + 1].strip()
        # skip the prediction line (i+2) — present but not used
        records.append((acc, seq))
        i += 3
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute AIUPred energy embeddings for all sequences"
    )
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--seq-dir",
        type=str,
        default=None,
        help="Directory containing per-protein FASTA files (one sequence per file)",
    )
    mode_group.add_argument(
        "--fasta",
        type=str,
        default=None,
        help="Combined fasta file (format: >ACC / SEQUENCE / PREDICTION per record)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/energy_embeddings",
        help="Output directory for energy embedding .npy files",
    )

    args = parser.parse_args()

    # Default to seq-dir mode if neither flag is given
    if args.seq_dir is None and args.fasta is None:
        args.seq_dir = "iupred2a/data/seq"

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize AIUPred predictor (loads model weights once)
    print("Initializing AIUPred predictor...")
    predictor = AIUPred()

    # Build list of (accession, sequence) pairs depending on mode
    if args.fasta is not None:
        print(f"Mode: combined fasta  ({args.fasta})")
        pairs = read_combined_fasta(args.fasta)
        print(f"Records in fasta: {len(pairs)}")
    else:
        print(f"Mode: per-protein fasta files  ({args.seq_dir})")
        seq_dir = Path(args.seq_dir)
        fasta_files = sorted(seq_dir.glob("*.fasta"))
        print(f"Found {len(fasta_files)} FASTA files in {args.seq_dir}")
        pairs = []
        for fasta_path in fasta_files:
            accession = fasta_path.stem
            fasta_dict = read_fasta(str(fasta_path))
            if not fasta_dict:
                print(f"Warning: Empty FASTA for {accession}")
                continue
            pairs.append((accession, list(fasta_dict.values())[0]))

    for accession, sequence in tqdm(pairs, desc="Computing energy embeddings"):
        output_path = Path(args.output_dir) / f"{accession}.npy"
        if output_path.exists():
            continue

        try:
            if not sequence:
                print(f"Warning: Empty sequence for {accession}")
                continue

            # Get center-only embedding: shape (L, 32)
            embedding = predictor.get_embedding(sequence, center_only=True)

            np.save(output_path, embedding)

        except Exception as e:
            print(f"Error processing {accession}: {e}")
            continue

    print(f"Done! Energy embeddings saved to {args.output_dir}")


if __name__ == "__main__":
    main()
