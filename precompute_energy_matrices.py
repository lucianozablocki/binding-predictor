#!/usr/bin/env python
"""
Pre-compute expanded energy matrices for all sequences.

This script reads the 20x20 amino acid energy matrix and all protein sequences,
then generates an LxL expanded energy matrix for each sequence where position (i,j)
contains the interaction energy between amino acid at position i and position j.

Usage:
    python scripts/precompute_energy_matrices.py \
        --energy-matrix data/iupred2_short_energy_matrix \
        --seq-dir iupred2a/data/seq \
        --output-dir data/expanded_energy_matrices
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str("src"))
from helper_functions import read_fasta

# Amino acid vocabulary
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def read_energy_matrix(filepath: str) -> np.ndarray:
    """
    Read an energy matrix file and convert it to a numpy matrix.
    
    Parameters
    ----------
    filepath : str
        Path to the energy matrix file.
        
    Returns
    -------
    np.ndarray
        A 20x20 numpy matrix.
    """
    n = len(AMINO_ACIDS)
    matrix = np.zeros((n, n), dtype=np.float32)
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            
            aa1, aa2, value = parts[0], parts[1], float(parts[2])
            
            if aa1 in AA_TO_INDEX and aa2 in AA_TO_INDEX:
                i = AA_TO_INDEX[aa1]
                j = AA_TO_INDEX[aa2]
                matrix[i, j] = value
    
    return matrix


def expand_energy_matrix(sequence: str, energy_matrix: np.ndarray) -> np.ndarray:
    """
    Create an LxL expanded energy matrix for a given sequence.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence string
    energy_matrix : np.ndarray
        20x20 energy matrix
        
    Returns
    -------
    np.ndarray
        LxL expanded energy matrix where L = len(sequence)
    """
    L = len(sequence)
    expanded = np.zeros((L, L), dtype=np.float32)
    
    # Convert sequence to amino acid indices
    aa_indices = []
    for aa in sequence:
        if aa in AA_TO_INDEX:
            aa_indices.append(AA_TO_INDEX[aa])
        else:
            # aa_indices.append(0)  # Default to first AA for unknown
            raise Exception(f"Unknown amino acid encountered: {aa}")

    # Fill the expanded matrix
    for i in range(L):
        for j in range(L):
            expanded[i, j] = energy_matrix[aa_indices[i], aa_indices[j]]
    
    return expanded


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute expanded energy matrices for all sequences"
    )
    parser.add_argument(
        "--energy-matrix", 
        type=str, 
        default="iupred2a/data/iupred2_short_energy_matrix",
        help="Path to the 20x20 energy matrix file"
    )
    parser.add_argument(
        "--seq-dir", 
        type=str, 
        default="iupred2a/data/seq",
        help="Directory containing FASTA sequence files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/expanded_energy_matrices",
        help="Output directory for expanded matrices"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["npy", "npz"],
        default="npy",
        help="Output format (npy for individual files, npz for single archive)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the energy matrix
    print(f"Reading energy matrix from {args.energy_matrix}")
    energy_matrix = read_energy_matrix(args.energy_matrix)
    print(f"Energy matrix shape: {energy_matrix.shape}")
    
    # Get all FASTA files
    seq_dir = Path(args.seq_dir)
    fasta_files = list(seq_dir.glob("*.fasta"))
    print(f"Found {len(fasta_files)} FASTA files in {args.seq_dir}")
    
    if args.format == "npz":
        # Store all matrices in a single archive
        all_matrices = {}
        
    # Process each sequence
    for fasta_path in tqdm(fasta_files, desc="Processing sequences"):
        accession = fasta_path.stem  # filename without extension
        
        try:
            fasta_dict = read_fasta(str(fasta_path))
            if not fasta_dict:
                print(f"Warning: Empty FASTA for {accession}")
                continue
                
            sequence = list(fasta_dict.values())[0]
            
            # Generate expanded matrix
            expanded = expand_energy_matrix(sequence, energy_matrix)
            
            if args.format == "npy":
                # Save individual file
                output_path = Path(args.output_dir) / f"{accession}.npy"
                # np.save(output_path, expanded)
            else:
                all_matrices[accession] = expanded
                
        except Exception as e:
            print(f"Error processing {accession}: {e}")
            continue
    
    if args.format == "npz":
        # Save single archive
        output_path = Path(args.output_dir) / "all_energy_matrices.npz"
        np.savez_compressed(output_path, **all_matrices)
        print(f"Saved all matrices to {output_path}")
    
    print(f"Done! Expanded matrices saved to {args.output_dir}")


if __name__ == "__main__":
    main()
