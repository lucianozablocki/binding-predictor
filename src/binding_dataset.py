import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from helper_functions import read_fasta, setup_logger 

ESM2_REPR_PATH = "data/2312_disprot_esm2_representations.pt"

# Amino Acid vocabulary for one-hot encoding
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i for i, aa in enumerate(AA_VOCAB)}
NUM_AMINO_ACIDS = len(AA_VOCAB)  # 20
logger = setup_logger(__name__)


def one_hot_encode(sequence_str, aa_to_id=AA_TO_ID, num_classes=NUM_AMINO_ACIDS):
    """
    One-hot encode an amino acid sequence.
    
    Args:
        sequence_str: String of amino acids
        aa_to_id: Dictionary mapping amino acids to indices
        num_classes: Number of amino acid classes (20)
    
    Returns:
        Tensor of shape (seq_len, num_classes) with one-hot encoding
        raise when Unknown amino acids are found
    """
    indices = []
    for aa in sequence_str:
        if aa in aa_to_id:
            indices.append(aa_to_id[aa])
        else:
            # indices.append(-1)  # Mark unknown amino acids
            raise Exception(f"Unknown amino acid encountered: {aa}")
    
    # Create one-hot tensor
    one_hot = torch.zeros(len(sequence_str), num_classes, dtype=torch.float32)
    for i, idx in enumerate(indices):
        one_hot[i, idx] = 1.0
    
    return one_hot


class BindingDataset(Dataset):
    def __init__(self, tsv_file, seq_dir, zone_annotations=("disorder",), energy_emb_dir='data/energy_embeddings', aa_to_id=AA_TO_ID):
        self.data = []
        self.aa_to_id = aa_to_id
        self.energy_emb_dir = Path(energy_emb_dir)
        self.zone_annotations = {label.strip().lower() for label in zone_annotations}

        # Load ESM2 representations once
        logger.info("Loading ESM2 representations...")
        self.esm2_reps = torch.load(ESM2_REPR_PATH, weights_only=True)

        # 1. Aggregate regions by Accession ID to handle multiple sites per protein
        # Mapping: accession -> list of (start, end) tuples
        accession_regions = defaultdict(list)
        accession_zone_regions = defaultdict(list)
        
        logger.info("Parsing TSV file...")
        with open(tsv_file, 'r') as fn:
            for line in tqdm(fn):
                # Skip header or empty lines if necessary
                if not line.strip() or line.startswith('acc'): 
                    continue
                    
                parts = line.split('\t')
                annotation = parts[11].strip().lower() if len(parts) > 11 else ""
                
                # Check for protein binding annotation (Col 11)
                if annotation == 'protein binding':
                    accession = parts[0]
                    # Convert to int, handle 1-based indexing later
                    try:
                        start = int(parts[7])
                        end = int(parts[8])
                        accession_regions[accession].append((start, end))
                    except ValueError:
                        continue # Skip malformed lines

                if annotation in self.zone_annotations:
                    accession = parts[0]
                    try:
                        start = int(parts[7])
                        end = int(parts[8])
                        accession_zone_regions[accession].append((start, end))
                    except ValueError:
                        continue

        logger.info(f"Processing {len(accession_regions)} unique proteins...")
        
        # 2. Load Sequences and Build Masks
        for accession, regions in tqdm(accession_regions.items()):
            # logger.info("lucsi")
            fasta_path = f'{seq_dir}/{accession}.fasta'
            
            try:
                # helper function returns {header: sequence}
                fasta_dict = read_fasta(fasta_path) 
                if not fasta_dict: 
                    logger.error(f'{accession} fasta reading resulted in empty dct')
                    continue
                
                # Extract sequence (values) not header (keys)
                sequence_str = list(fasta_dict.values())[0]
                if len(sequence_str)>4000:
                    logger.info(f'Skipping {accession} due to length {len(sequence_str)}')
                    continue
                seq_len = len(sequence_str)
                
                # ESM2 representation (seq_len, 1280)
                if accession not in self.esm2_reps:
                    raise KeyError(f"Missing ESM2 representation for {accession}. Run generate_esm2_representations.py first.")
                encoded_seq = self.esm2_reps[accession]
                if encoded_seq.shape[0] != seq_len:
                    raise ValueError(
                        f"ESM2 representation length {encoded_seq.shape[0]} != sequence length {seq_len} for {accession}"
                    )
                # Build Target Mask (0 = background, 1 = binding)
                target_mask = torch.zeros((seq_len,), dtype=torch.float32)
                zone_mask = torch.zeros((seq_len,), dtype=torch.float32)
                
                for start, end in regions:
                    # DisProt is 1-based inclusive. 
                    # Python is 0-based exclusive.
                    # e.g., 1-3 means indices 0, 1, 2
                    
                    # Bounds check to prevent crashes if annotation exceeds sequence
                    s = max(0, start - 1)
                    e = min(seq_len, end)
                    
                    if s < seq_len:  # Really hope this does not happen...
                        target_mask[s:e] = 1.0
                    else:
                        logger.error(f'For some magical reason the start position is overindexed at {accession} pos {s}')

                for start, end in accession_zone_regions.get(accession, []):
                    s = max(0, start - 1)
                    e = min(seq_len, end)
                    if s < seq_len:
                        zone_mask[s:e] = 1.0
                    else:
                        logger.error(f'Zone label overindexed at {accession} pos {s}')
                
                # Load precomputed energy embedding (L, 32)
                emb_path = self.energy_emb_dir / f"{accession}.npy"
                if not emb_path.exists():
                    raise FileNotFoundError(
                        f"Missing energy embedding for {accession} at {emb_path}. "
                        f"Run precompute_energy_embeddings.py first."
                    )
                energy_emb = torch.from_numpy(np.load(emb_path)).float()
                if energy_emb.shape[0] != seq_len:
                    raise ValueError(
                        f"Energy embedding length {energy_emb.shape[0]} != sequence length {seq_len} for {accession}"
                    )

                self.data.append((encoded_seq, target_mask, zone_mask, accession, energy_emb))
                
            except FileNotFoundError as e:
                # print(f"Missing FASTA for {accession}")
                logger.error(f"Missing FASTA for {accession}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing {accession}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns tuple: (sequence_tensor, target_tensor)
        return self.data[idx]


def pad_collate(batch):
    """
    Pads sequences and targets to the longest length in the batch.
    Returns:
        padded_seqs: (Batch, Max_Len, NUM_AMINO_ACIDS) - one-hot encoded
        padded_targets: (Batch, Max_Len)
        lengths: (Batch) - useful for packing sequences or masking loss later
        accessions: tuple of accession IDs
        padded_energy_embs: (Batch, Max_Len, 32) - energy embeddings
    """
    (seqs, targets, zone_masks, accessions, energy_embs) = zip(*batch)
    

    # Calculate lengths (optional, but often useful for masking loss)
    lengths = torch.tensor([len(s) for s in seqs])
    
    # Pad sequences with zeros (seq_len, num_features) -> (batch, max_len, num_features)
    padded_seqs = pad_sequence(list(seqs), batch_first=True, padding_value=0.0)
    
    # Pad targets with 0 (background class) BE AWARE THIS MIGHT BE WRONG!
    padded_targets = pad_sequence(list(targets), batch_first=True, padding_value=-1)
    padded_zone_masks = pad_sequence(list(zone_masks), batch_first=True, padding_value=-1)
    
    # Pad energy embeddings with zeros (seq_len, 32) -> (batch, max_len, 32)
    padded_energy_embs = pad_sequence(energy_embs, batch_first=True, padding_value=0.0)
    
    return padded_seqs, padded_targets, padded_zone_masks, lengths, accessions, padded_energy_embs

def get_binding_dataloader(tsv_file, seq_dir, batch_size=32, shuffle=True, zone_annotations=("disorder",), energy_emb_dir='data/energy_embeddings'):
    dataset = BindingDataset(tsv_file, seq_dir, zone_annotations=zone_annotations, energy_emb_dir=energy_emb_dir)
    # logger.error(len(dataset))
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=pad_collate
    )
    
    return loader


class CSVBindingDataset(Dataset):
    """Dataset that reads (sequence, target, accession) from a CSV file.

    The CSV must have at least three columns:
        - ``accession``: UniProt-style accession ID (used to look up embeddings)
        - ``sequence``: Amino acid sequence string (standard 20 AAs)
        - ``target``: Per-residue binary labels as a string of '0'/'1' characters
                      with the same length as the sequence (e.g. "0010110")

    ESM2 representations and energy embeddings are loaded exactly as in
    :class:`BindingDataset`, so the same pre-computation steps are required.
    Zone masks are set to zeros (no zone information available from CSV).
    """

    def __init__(self, csv_file, esm2_repr_path=ESM2_REPR_PATH,
                 energy_emb_dir='data/energy_embeddings'):
        self.data = []
        self.energy_emb_dir = Path(energy_emb_dir)

        logger.info("Loading ESM2 representations...")
        esm2_reps = torch.load(esm2_repr_path, weights_only=True)

        logger.info(f"Reading CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        required_cols = {"accession", "sequence", "target"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            accession = str(row["accession"])
            sequence_str = str(row["sequence"])
            target_str = str(row["target"])
            seq_len = len(sequence_str)

            # if seq_len > 1000:
            #     logger.info(f"Skipping {accession} due to length {seq_len}")
            #     continue

            if len(target_str) != seq_len:
                raise ValueError(
                    f"{accession}: target length {len(target_str)} != "
                    f"sequence length {seq_len}. Check the preprocessing step that "
                    f"produced the target column."
                )

            # ESM2 representation
            if accession not in esm2_reps:
                raise KeyError(
                    f"{accession}: missing ESM2 representation in {esm2_repr_path}. "
                    f"Run generate_esm2_representations.py for this split first."
                )
            encoded_seq = esm2_reps[accession]
            if encoded_seq.shape[0] != seq_len:
                raise ValueError(
                    f"{accession}: ESM2 representation length {encoded_seq.shape[0]} != "
                    f"sequence length {seq_len}. The representation file may be stale."
                )

            # Target mask from CSV column
            try:
                target_mask = torch.tensor(
                    [float(c) for c in target_str], dtype=torch.float32
                )
            except ValueError as e:
                raise ValueError(
                    f"{accession}: target column contains non-numeric characters "
                    f"({e}). Expected a string of '0'/'1' chars."
                ) from e

            # Zone mask: zeros (no zone info in CSV)
            zone_mask = torch.zeros(seq_len, dtype=torch.float32)

            # Energy embedding
            emb_path = self.energy_emb_dir / f"{accession}.npy"
            if not emb_path.exists():
                raise FileNotFoundError(
                    f"{accession}: missing energy embedding at {emb_path}. "
                    f"Run precompute_energy_embeddings.py for this split first."
                )
            energy_emb = torch.from_numpy(np.load(emb_path)).float()
            if energy_emb.shape[0] != seq_len:
                raise ValueError(
                    f"{accession}: energy embedding length {energy_emb.shape[0]} != "
                    f"sequence length {seq_len}. The embedding file may be stale."
                )

            self.data.append((encoded_seq, target_mask, zone_mask, accession, energy_emb))

        logger.info(f"Loaded {len(self.data)} proteins from {csv_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_csv_binding_dataloader(csv_file, esm2_repr_path=ESM2_REPR_PATH,
                                batch_size=32, shuffle=False,
                                energy_emb_dir='data/energy_embeddings'):
    dataset = CSVBindingDataset(csv_file, esm2_repr_path=esm2_repr_path,
                                energy_emb_dir=energy_emb_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=pad_collate)

# --- Run ---
if __name__ == "__main__":
    # Create the loader
    train_loader = get_binding_dataloader(
        # tsv_file='iupred2a/data/disprot_v_25_06.tsv',
        tsv_file='iupred2a/data/DisProt_2023_12_IDPO-GO.tsv',
        seq_dir='iupred2a/data/seq'
    )
    
    # Iterate through one batch to verify
    for seqs, targets, zone_masks, lengths, accessions, energy_embs in train_loader:
        print(f"Batch Shape Inputs: {seqs.shape}")   # [32, MAX_LEN, 20]
        print(f"Batch Shape Targets: {targets.shape}") # [32, MAX_LEN]
        print(f"Batch Shape Zones: {zone_masks.shape}") # [32, MAX_LEN]
        print(f"First Sequence Length: {lengths[0]}")
        print(f"First position one-hot: {seqs[0, 0]}")  # Should be one-hot vector
        print(f"Len of first position one-hot: {len(seqs[0])}")
        print(f"Sum of first position (should be 1 for known AA): {seqs[0, 0].sum()}")
        print(targets[0])
        break
