import torch
import numpy as np

matrix_path = "iupred2a/data/iupred2_short_energy_matrix"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def outer_concat(t1: torch.Tensor, t2: torch.Tensor):
    # t1, t2: shape = B x L x E
    assert t1.shape == t2.shape, f"Shapes of input tensors must match! ({t1.shape} != {t2.shape})"

    seq_len = t1.shape[1]
    a = t1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
    b = t2.unsqueeze(-3).expand(-1, seq_len, -1, -1)

    return torch.concat((a, b), dim=-1)

def mat2bp(x):
    """Get base-pairs from conection matrix [N, N]. It uses upper
    triangular matrix only, without the diagonal. Positions are 1-based. """
    ind = torch.triu_indices(x.shape[0], x.shape[1], offset=1)
    pairs_ind = torch.where(x[ind[0], ind[1]] > 0)[0]

    pairs_ind = ind[:, pairs_ind].T
    # remove multiplets pairs
    multiplets = []
    for i, j in pairs_ind:
        ind = torch.where(pairs_ind[:, 1]==i)[0]
        if len(ind)>0:
            pairs = [bp.tolist() for bp in pairs_ind[ind]] + [[i.item(), j.item()]]
            best_pair = torch.tensor([x[bp[0], bp[1]] for bp in pairs]).argmax()
                
            multiplets += [pairs[k] for k in range(len(pairs)) if k!=best_pair]   
            
    pairs_ind = [[bp[0]+1, bp[1]+1] for bp in pairs_ind.tolist() if bp not in multiplets]
 
    return pairs_ind

def bp2matrix(L, base_pairs):
    matrix = torch.zeros((L, L))
    # base pairs are 1-based
    bp = torch.tensor(base_pairs) - 1
    if len(bp.shape) == 2:
        matrix[bp[:, 0], bp[:, 1]] = 1
        matrix[bp[:, 1], bp[:, 0]] = 1

    return matrix

def get_embed_dim(loader):
    # grab an element from the loader, which is represented by a dictionary with keys
    # `seq_ids`, `seq_embs_pad`, `contacts`, `Ls`
    batch_elem = next(iter(loader))
    # query for `seq_embs_pad` key (containing the embedding representations of all the sequences in the batch)
    # whose size will be batch_size x L x d
    return batch_elem[0].shape[-1]

def read_energy_matrix(filepath: str) -> np.ndarray:
    """
    Read an energy matrix file and convert it to a numpy matrix.
    
    The file format is expected to be:
        AA1 AA2 value
    where AA1 and AA2 are single-letter amino acid codes.
    
    Parameters
    ----------
    filepath : str
        Path to the energy matrix file.
        
    Returns
    -------
    np.ndarray
        A 20x20 numpy matrix where entry [i,j] contains the energy
        value for the amino acid pair (AMINO_ACIDS[i], AMINO_ACIDS[j]).
    """
    n = len(AMINO_ACIDS)
    matrix = np.zeros((n, n), dtype=np.float64)
    
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

def expand_energy_matrix(embedding, energy_matrix_path=matrix_path):
    """
    Expand a 20x20 amino acid energy matrix to LxL based on sequence positions.
    
    Given a one-hot encoded sequence embedding (B, L, 20) and an energy matrix (20, 20),
    creates an expanded matrix (B, L, L) where position [b, i, j] contains the
    interaction energy between the amino acid at position i and position j.
    
    Parameters
    ----------
    embedding : torch.Tensor
        One-hot encoded sequences of shape (B, L, 20)
    energy_matrix_path : str
        Path to the energy matrix file
        
    Returns
    -------
    torch.Tensor
        Expanded energy matrix of shape (B, L, L)
    """
    # Read energy matrix (20x20)
    energy_matrix = read_energy_matrix(energy_matrix_path)
    
    B, L, E = embedding.shape
    expanded_energy_matrix = torch.zeros((B, L, L), dtype=embedding.dtype, device=embedding.device)
    
    # Convert one-hot to amino acid indices for each position
    # argmax gives the index of the 1 in each one-hot vector
    aa_indices = embedding.argmax(dim=-1)  # (B, L)
    
    for b in range(B):
        for i in range(L):
            for j in range(L):
                aa_i = aa_indices[b, i].item()
                aa_j = aa_indices[b, j].item()
                expanded_energy_matrix[b, i, j] = energy_matrix[aa_i, aa_j]
    
    return expanded_energy_matrix
