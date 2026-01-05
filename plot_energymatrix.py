import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Standard amino acid single-letter codes
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Create mapping from amino acid letter to index
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


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


def get_amino_acid_labels() -> list:
    """Return the list of amino acids in order (for labeling axes)."""
    return AMINO_ACIDS.copy()


def get_aa_index(aa: str) -> int:
    """Get the matrix index for an amino acid."""
    return AA_TO_INDEX.get(aa.upper(), -1)


def plot_energy_matrix(matrix: np.ndarray, title: str = "Energy Matrix", 
                       cmap: str = "RdBu_r", figsize: tuple = (12, 10),
                       save_path: str = None) -> None:
    """
    Plot the energy matrix as a colored heatmap.
    
    Parameters
    ----------
    matrix : np.ndarray
        A 20x20 energy matrix.
    title : str
        Title for the plot.
    cmap : str
        Colormap to use (default: RdBu_r - red for positive, blue for negative).
    figsize : tuple
        Figure size in inches.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with symmetric color scale around 0
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    vmin = -vmax
    
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(AMINO_ACIDS)))
    ax.set_yticks(np.arange(len(AMINO_ACIDS)))
    ax.set_xticklabels(AMINO_ACIDS, fontsize=10)
    ax.set_yticklabels(AMINO_ACIDS, fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Energy Value", rotation=-90, va="bottom", fontsize=12)
    
    # Labels and title
    ax.set_xlabel("Amino Acid", fontsize=12)
    ax.set_ylabel("Amino Acid", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add grid lines
    ax.set_xticks(np.arange(len(AMINO_ACIDS) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(AMINO_ACIDS) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage with the short energy matrix
    matrix_path = Path(__file__).parent / "iupred2a" / "data" / "iupred2_short_energy_matrix"
    
    matrix = read_energy_matrix(matrix_path)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Amino acid order: {AMINO_ACIDS}")
    print(f"\nExample: F-F energy = {matrix[AA_TO_INDEX['F'], AA_TO_INDEX['F']]:.4f}")
    print(f"Example: C-C energy = {matrix[AA_TO_INDEX['C'], AA_TO_INDEX['C']]:.4f}")
    
    # Plot the energy matrix
    plot_energy_matrix(matrix, title="IUPred2 Short Energy Matrix")
