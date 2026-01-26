#!/usr/bin/env python3
"""
Script to randomly split a DisProt TSV file into train and validation sets.
Splits by unique protein accessions to avoid data leakage.
"""

import argparse
import random
from pathlib import Path


def split_data(input_file: str, output_dir: str, val_ratio: float = 0.2, seed: int = 42):
    """
    Randomly split a TSV file into train and validation sets by protein accession.
    
    Args:
        input_file: Path to the input TSV file
        output_dir: Directory where train.tsv and val.tsv will be created
        val_ratio: Fraction of proteins to use for validation (default: 0.2)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read the TSV file
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print(f"No data found in {input_file}")
        return
    
    # Separate header and data
    header = lines[0]
    data_lines = lines[1:]
    
    # Group lines by accession (first column)
    accession_to_lines = {}
    for line in data_lines:
        if not line.strip():
            continue
        accession = line.split('\t')[0]
        if accession not in accession_to_lines:
            accession_to_lines[accession] = []
        accession_to_lines[accession].append(line)
    
    # Get unique accessions and shuffle
    accessions = list(accession_to_lines.keys())
    random.shuffle(accessions)
    
    print(f"Found {len(accessions)} unique protein accessions")
    print(f"Total rows: {len(data_lines)}")
    
    # Calculate split
    val_size = int(len(accessions) * val_ratio)
    
    val_accessions = set(accessions[:val_size])
    train_accessions = set(accessions[val_size:])
    
    # Collect lines for each split
    train_lines = []
    val_lines = []
    
    for acc in train_accessions:
        train_lines.extend(accession_to_lines[acc])
    
    for acc in val_accessions:
        val_lines.extend(accession_to_lines[acc])
    
    print(f"\nTrain set: {len(train_accessions)} proteins, {len(train_lines)} rows")
    print(f"Validation set: {len(val_accessions)} proteins, {len(val_lines)} rows")
    
    # Write output files
    train_file = output_path / "train.tsv"
    val_file = output_path / "val.tsv"
    
    with open(train_file, 'w') as f:
        f.write(header)
        f.writelines(train_lines)
    
    with open(val_file, 'w') as f:
        f.write(header)
        f.writelines(val_lines)
    
    print(f"\nDone! Files saved to:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")


def main():
    parser = argparse.ArgumentParser(description="Split DisProt TSV file into train and validation sets")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="iupred2a/data/disprot_v_25_06.tsv",
        help="Input TSV file (default: iupred2a/data/disprot_v_25_06.tsv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="iupred2a/data",
        help="Output directory for train.tsv and val.tsv (default: iupred2a/data)"
    )
    parser.add_argument(
        "--val-ratio", "-v",
        type=float,
        default=0.2,
        help="Fraction of proteins for validation (default: 0.2)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    split_data(
        input_file=args.input,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
