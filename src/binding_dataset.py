import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from tqdm import tqdm
from helper_functions import read_fasta, setup_logger 

# Simple dictionary to map Amino Acids to Integers
# 0 is reserved for Padding, so we start at 1
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}
logger = setup_logger(__name__)


class BindingDataset(Dataset):
    def __init__(self, tsv_file, seq_dir, aa_to_id=AA_TO_ID):
        self.data = []
        self.aa_to_id = aa_to_id
        
        # 1. Aggregate regions by Accession ID to handle multiple sites per protein
        # Mapping: accession -> list of (start, end) tuples
        accession_regions = defaultdict(list)
        
        logger.info("Parsing TSV file...")
        with open(tsv_file, 'r') as fn:
            for line in tqdm(fn):
                # Skip header or empty lines if necessary
                if not line.strip() or line.startswith('acc'): 
                    continue
                    
                parts = line.split('\t')
                
                # Check for protein binding annotation (Col 11)
                if len(parts) > 11 and parts[11] == 'protein binding':
                    accession = parts[0]
                    # Convert to int, handle 1-based indexing later
                    try:
                        start = int(parts[7])
                        end = int(parts[8])
                        accession_regions[accession].append((start, end))
                    except ValueError:
                        continue # Skip malformed lines

        logger.info(f"Processing {len(accession_regions)} unique proteins...")
        
        # 2. Load Sequences and Build Masks
        for accession, regions in tqdm(accession_regions.items()):
            fasta_path = f'{seq_dir}/{accession}.fasta'
            
            try:
                # helper function returns {header: sequence}
                fasta_dict = read_fasta(fasta_path) 
                if not fasta_dict: 
                    logger.error(f'{accession} fasta reading resulted in empty dct')
                    continue
                
                # Extract sequence (values) not header (keys)
                sequence_str = list(fasta_dict.values())[0]
                seq_len = len(sequence_str)
                
                # Tokenize sequence (Map chars to ints, default to 0 for unknown)
                tokenized_seq = torch.tensor(
                    [self.aa_to_id.get(aa, 0) for aa in sequence_str], 
                    dtype=torch.long
                )
                
                # Build Target Mask (0 = background, 1 = binding)
                target_mask = torch.zeros((seq_len,), dtype=torch.float32)
                
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
                
                self.data.append((tokenized_seq, target_mask))
                
            except FileNotFoundError:
                # print(f"Missing FASTA for {accession}")
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
        padded_seqs: (Batch, Max_Len)
        padded_targets: (Batch, Max_Len)
        lengths: (Batch) - useful for packing sequences or masking loss later
    """
    (seqs, targets) = zip(*batch)
    
    # Calculate lengths (optional, but often useful for masking loss)
    lengths = torch.tensor([len(s) for s in seqs])
    
    # Pad sequences with 0 
    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    
    # Pad targets with 0 (background class) BE AWARE THIS MIGHT BE WRONG!
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return padded_seqs, padded_targets, lengths


def get_binding_dataloader(tsv_file, seq_dir, batch_size=32, shuffle=True):
    dataset = BindingDataset(tsv_file, seq_dir)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=pad_collate
    )
    
    return loader

# --- Run ---
if __name__ == "__main__":
    # Create the loader
    train_loader = get_binding_dataloader(
        tsv_file='iupred2a/data/disprot_v_25_06.tsv',
        seq_dir='iupred2a/data/seq'
    )
    
    # Iterate through one batch to verify
    for seqs, targets, lengths in train_loader:
        print(f"Batch Shape Inputs: {seqs.shape}")   # [32, MAX_LEN]
        print(f"Batch Shape Targets: {targets.shape}") # [32, MAX_LEN]
        print(f"First Sequence Length: {lengths[0]}")
        print(seqs[0])
        print(targets[0])
        break
