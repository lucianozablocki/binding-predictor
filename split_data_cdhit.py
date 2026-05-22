#!/usr/bin/env python3
"""
Split a DisProt TSV file into train and validation sets based on CD-HIT
sequence clustering. Entire clusters are assigned to one split, ensuring
no test protein shares high sequence identity with any training protein.
"""

import argparse
import random
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


def build_combined_fasta(seq_dir: str, accessions: list[str], output_fasta: str,
                         max_length: int = 1000):
    """Concatenate per-accession FASTA files into a single FASTA, skipping
    sequences longer than max_length."""
    seq_path = Path(seq_dir)
    found = 0
    skipped_long = 0
    with open(output_fasta, "w") as out:
        for acc in accessions:
            fasta_file = seq_path / f"{acc}.fasta"
            if not fasta_file.exists():
                continue
            with open(fasta_file) as f:
                content = f.read()
            # Rewrite header to just the accession for easy parsing
            lines = content.strip().split("\n")
            seq = "".join(line for line in lines if not line.startswith(">"))
            if len(seq) > max_length:
                skipped_long += 1
                continue
            out.write(f">{acc}\n")
            out.write(seq + "\n")
            found += 1
    print(f"Wrote {found}/{len(accessions)} sequences to {output_fasta}"
          f" (skipped {skipped_long} longer than {max_length})")
    return found


def run_cdhit(input_fasta: str, output_prefix: str,
              identity_threshold: float = 0.4, word_size: int = 2,
              threads: int = 0, memory_mb: int = 30000) -> str:
    """Run CD-HIT and return path to the .clstr file."""
    if shutil.which("cd-hit") is None:
        raise RuntimeError(
            "cd-hit not found in PATH. Install it with:\n"
            "  conda install -c bioconda cd-hit\n"
            "or\n"
            "  apt install cd-hit"
        )

    cmd = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_prefix,
        "-c", str(identity_threshold),
        "-n", str(word_size),
        "-T", str(threads),
        "-M", str(memory_mb),
    ]
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    clstr_file = f"{output_prefix}.clstr"
    if not Path(clstr_file).exists():
        raise FileNotFoundError(f"Expected cluster file not found: {clstr_file}")
    return clstr_file


def parse_clstr(clstr_file: str) -> dict[int, list[str]]:
    """
    Parse a CD-HIT .clstr file.
    Returns: {cluster_id: [accession1, accession2, ...]}
    """
    clusters: dict[int, list[str]] = {}
    current_cluster = -1
    acc_pattern = re.compile(r">(.+?)\.\.\.")

    with open(clstr_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[-1])
                clusters[current_cluster] = []
            else:
                match = acc_pattern.search(line)
                if match:
                    clusters[current_cluster].append(match.group(1))
    return clusters


def assign_clusters_to_splits(
    clusters: dict[int, list[str]],
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """
    Assign whole clusters to train or val, targeting val_ratio by protein count.
    Clusters are sorted largest-first; each cluster is greedily placed into the
    split that is furthest below its target count.
    """
    random.seed(seed)
    total_proteins = sum(len(members) for members in clusters.values())
    val_target = int(total_proteins * val_ratio)

    # Step 1: Shuffle cluster IDs so that clusters of the same size
    # are ordered randomly (controlled by seed for reproducibility)
    cluster_ids = list(clusters.keys())
    random.shuffle(cluster_ids)

    # Step 2: Sort by cluster size, largest first. This is a greedy
    # bin-packing trick — placing big clusters first avoids overshooting
    # the val target at the end with a large cluster.
    cluster_ids.sort(key=lambda cid: len(clusters[cid]), reverse=True)

    train_acc: set[str] = set()
    val_acc: set[str] = set()
    val_count = 0

    # Step 3: Walk through clusters one by one. If adding the cluster
    # to val doesn't exceed the target, put it in val; otherwise train.
    for cid in cluster_ids:
        members = clusters[cid]
        if val_count + len(members) <= val_target:
            val_acc.update(members)
            val_count += len(members)
        else:
            train_acc.update(members)

    return train_acc, val_acc


def evaluate_split(train_fasta: str, val_fasta: str, output_prefix: str,
                   identity_threshold: float = 0.4, word_size: int = 2):
    """
    Use cd-hit-2d to find val proteins that are similar to train proteins.
    Reports how many val sequences exceed the identity threshold and lists them.
    """
    if shutil.which("cd-hit-2d") is None:
        raise RuntimeError(
            "cd-hit-2d not found in PATH. Install it with:\n"
            "  conda install -c bioconda cd-hit"
        )

    cmd = [
        "cd-hit-2d",
        "-i", train_fasta,   # database (train)
        "-i2", val_fasta,    # query (val)
        "-o", output_prefix,
        "-c", str(identity_threshold),
        "-n", str(word_size),
        "-T", "0",
        "-M", "30000",
    ]
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    # Parse the .clstr file: clusters with >1 member contain cross-set matches
    clstr_file = f"{output_prefix}.clstr"
    clusters = parse_clstr(clstr_file)

    leaky_pairs = []  # (val_acc, train_acc) pairs that are too similar
    for members in clusters.values():
        if len(members) > 1:
            # In cd-hit-2d clusters, the representative is from train (db1)
            # and non-representatives are from val (db2) that matched it
            leaky_pairs.extend(members)

    # Count val sequences in the output FASTA (these are the ones that
    # did NOT match any train sequence above the threshold)
    non_redundant = 0
    with open(output_prefix) as f:
        for line in f:
            if line.startswith(">"):
                non_redundant += 1

    # Count total val sequences
    total_val = 0
    with open(val_fasta) as f:
        for line in f:
            if line.startswith(">"):
                total_val += 1

    matched = total_val - non_redundant
    print(f"\n=== Split Evaluation (identity threshold: {identity_threshold}) ===")
    print(f"Val sequences: {total_val}")
    print(f"Val sequences similar to a train sequence: {matched}")
    print(f"Val sequences with no train match: {non_redundant}")
    if matched > 0:
        print(f"\nWARNING: {matched} val protein(s) share >{identity_threshold*100:.0f}% "
              f"identity with a train protein — potential leakage.")
    else:
        print(f"\nNo val protein shares >{identity_threshold*100:.0f}% identity "
              f"with any train protein. Split looks clean.")


def split_tsv(input_tsv: str, output_dir: str,
              train_acc: set[str], val_acc: set[str]):
    """Write train.tsv and val.tsv based on accession sets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(input_tsv) as f:
        lines = f.readlines()

    header = lines[0]
    train_lines = []
    val_lines = []
    skipped = 0

    for line in lines[1:]:
        if not line.strip():
            continue
        acc = line.split("\t")[0]
        if acc in train_acc:
            train_lines.append(line)
        elif acc in val_acc:
            val_lines.append(line)
        else:
            skipped += 1

    train_file = output_path / "train.tsv"
    val_file = output_path / "val.tsv"

    with open(train_file, "w") as f:
        f.write(header)
        f.writelines(train_lines)

    with open(val_file, "w") as f:
        f.write(header)
        f.writelines(val_lines)

    print(f"\nTrain: {len(train_acc)} proteins, {len(train_lines)} rows -> {train_file}")
    print(f"Val:   {len(val_acc)} proteins, {len(val_lines)} rows -> {val_file}")
    if skipped:
        print(f"Skipped {skipped} rows (accession not in any cluster / no FASTA)")


def main():
    parser = argparse.ArgumentParser(
        description="Split DisProt TSV into train/val by CD-HIT sequence clusters"
    )
    parser.add_argument(
        "--input", "-i", default="iupred2a/data/disprot_v_25_06.tsv",
        help="Input TSV file (default: iupred2a/data/disprot_v_25_06.tsv)",
    )
    parser.add_argument(
        "--seq-dir", "-d", default="iupred2a/data/seq",
        help="Directory with per-accession FASTA files",
    )
    parser.add_argument(
        "--output", "-o", default="iupred2a/data",
        help="Output directory for train.tsv and val.tsv",
    )
    parser.add_argument(
        "--identity", "-c", type=float, default=0.4,
        help="CD-HIT sequence identity threshold (default: 0.4 = 40%%)",
    )
    parser.add_argument(
        "--word-size", "-n", type=int, default=2,
        help="CD-HIT word size (must match identity; default: 2 for <=0.7)",
    )
    parser.add_argument(
        "--val-ratio", "-v", type=float, default=0.2,
        help="Target fraction of proteins for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-length", "-l", type=int, default=1000,
        help="Skip sequences longer than this (default: 1000, matching BindingDataset)",
    )
    parser.add_argument(
        "--keep-tmp", action="store_true",
        help="Keep temporary CD-HIT files instead of cleaning up",
    )
    parser.add_argument(
        "--evaluate", "-e", action="store_true",
        help="Evaluate an existing train/val split instead of creating a new one. "
             "Reads train.tsv and val.tsv from --output dir and reports cross-set similarity.",
    )
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="cdhit_split_")

    if args.evaluate:
        # --- Evaluate an existing split ---
        # Read accessions from existing train.tsv and val.tsv
        output_path = Path(args.output)
        train_tsv = output_path / "train.tsv"
        val_tsv = output_path / "val.tsv"
        if not train_tsv.exists() or not val_tsv.exists():
            print(f"Error: {train_tsv} and/or {val_tsv} not found. Run split first.")
            return

        def accessions_from_tsv(tsv_path):
            with open(tsv_path) as f:
                return sorted({line.split('\t')[0] for line in f.readlines()[1:] if line.strip()})

        train_accs = accessions_from_tsv(train_tsv)
        val_accs = accessions_from_tsv(val_tsv)
        print(f"Existing split: {len(train_accs)} train, {len(val_accs)} val proteins")

        # Build separate FASTA files for train and val
        train_fasta = str(Path(tmp_dir) / "train.fasta")
        val_fasta = str(Path(tmp_dir) / "val.fasta")
        build_combined_fasta(args.seq_dir, train_accs, train_fasta, max_length=args.max_length)
        build_combined_fasta(args.seq_dir, val_accs, val_fasta, max_length=args.max_length)

        # Run cd-hit-2d to find cross-set similarities
        cdhit2d_out = str(Path(tmp_dir) / "cdhit2d_out")
        evaluate_split(train_fasta, val_fasta, cdhit2d_out,
                       identity_threshold=args.identity, word_size=args.word_size)
    else:
        # --- Create a new cluster-based split ---
        # 1. Collect accessions from the TSV
        with open(args.input) as f:
            lines = f.readlines()
        accessions = sorted({
            line.split("\t")[0]
            for line in lines[1:]
            if line.strip()
        })
        print(f"Found {len(accessions)} unique accessions in {args.input}")

        # 2. Build combined FASTA and run CD-HIT
        combined_fasta = str(Path(tmp_dir) / "all_proteins.fasta")
        cdhit_out = str(Path(tmp_dir) / "cdhit_out")

        build_combined_fasta(args.seq_dir, accessions, combined_fasta,
                              max_length=args.max_length)

        clstr_file = run_cdhit(
            combined_fasta, cdhit_out,
            identity_threshold=args.identity,
            word_size=args.word_size,
        )

        # 3. Parse clusters
        clusters = parse_clstr(clstr_file)
        sizes = [len(m) for m in clusters.values()]
        clustered_proteins = sum(sizes)
        print(f"\nCD-HIT found {len(clusters)} clusters covering {clustered_proteins} proteins")
        print(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
              f"median={sorted(sizes)[len(sizes)//2]}, singletons={sizes.count(1)}")

        # 4. Split clusters
        train_acc, val_acc = assign_clusters_to_splits(
            clusters, args.val_ratio, args.seed
        )

        # 5. Write train/val TSVs
        split_tsv(args.input, args.output, train_acc, val_acc)

    # Cleanup
    if args.keep_tmp:
        print(f"\nTemporary files kept in: {tmp_dir}")
    else:
        import shutil as _shutil
        _shutil.rmtree(tmp_dir)
        print(f"\nTemporary files cleaned up.")


if __name__ == "__main__":
    main()
