import os
import torch
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# MODE = "tsv"    : read accessions from TSV, sequences from per-protein fasta files
# MODE = "fasta"  : read accessions and sequences from a single combined fasta
#                   (format: >ACC / SEQUENCE / PREDICTION  — one record = 3 lines)
MODE = "tsv"  # change to "fasta" to use the combined-fasta code path

# TSV mode config
TSV_PATH   = "iupred2a/data/disprot_v_25_06.tsv"
SEQ_DIR    = "iupred2a/data/seq"

# Combined-fasta mode config
FASTA_PATH = "iupred2a/data/caid3/binding.fasta"

OUTPUT_PATH = "/content/drive/MyDrive/esm2_representations/caid3binding_esm2_representations.pt"   # change as needed
BATCH_SIZE = 8       # reduce if you hit OOM
REPR_LAYER = 33      # last layer of esm2_t33_650M_UR50D
MAX_SEQ_LEN = 1000   # ESM2 positional encoding limit; longer seqs are truncated

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Running on {device}")

# ── Parse sequences ───────────────────────────────────────────────────────────
def read_fasta(path: str) -> str | None:
    """Return the full (concatenated) sequence from a single-record fasta file."""
    lines = Path(path).read_text().splitlines()
    seq_lines = [l.strip() for l in lines if l.strip() and not l.startswith(">")]
    if not seq_lines:
        return None
    return "".join(seq_lines)


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


data = []   # list of (acc, sequence)
skipped = []

if MODE == "fasta":
    # ── Combined-fasta code path ──────────────────────────────────────────────
    print(f"Mode: combined fasta  ({FASTA_PATH})")
    raw = read_combined_fasta(FASTA_PATH)
    print(f"Records in fasta: {len(raw)}")
    for acc, seq in raw:
        if not seq:
            skipped.append((acc, "empty sequence"))
            print(f"{acc} empty sequence")
            continue
        if len(seq) > MAX_SEQ_LEN:
            print(f"[warn] {acc}: length {len(seq)} > {MAX_SEQ_LEN}, skipping")
            skipped.append((acc, "too long"))
            continue
        data.append((acc, seq))

elif MODE == "tsv":
    # ── TSV + per-protein fasta code path ────────────────────────────────────
    print(f"Mode: TSV  ({TSV_PATH})")
    df = pd.read_csv(TSV_PATH, sep="\t", usecols=["acc", "term_name"])
    # Using standard equality operator for string matching
    protein_binding_accessions = df[df["term_name"] == "protein binding"]["acc"].dropna().unique().tolist()
    print(f"Unique accessions in TSV: {len(protein_binding_accessions)}")
    for acc in protein_binding_accessions:
        fasta_path = os.path.join(SEQ_DIR, f"{acc}.fasta")
        if not os.path.isfile(fasta_path):
            skipped.append((acc, "file not found"))
            print(f"{acc} file not found")
            continue
        seq = read_fasta(fasta_path)
        if not seq:
            skipped.append((acc, "empty sequence"))
            print(f"{acc} empty sequence")
            continue
        if len(seq) > MAX_SEQ_LEN:
            print(f"[warn] {acc}: length {len(seq)} > {MAX_SEQ_LEN}, skipping")
            skipped.append((acc, "too long"))
            continue
        data.append((acc, seq))

else:
    raise ValueError(f"Unknown MODE {MODE!r}. Choose 'tsv' or 'fasta'.")

print(f"Sequences to embed : {len(data)}")
print(f"Skipped            : {len(skipped)}")
if skipped:
    print("  " + "\n  ".join(f"{a}: {r}" for a, r in skipped[:10]))

# ── Process in batches ────────────────────────────────────────────────────────
# output_path = os.path.join(OUTPUT_DIR, "esm2_representations.pt")

# Load already-processed accessions so we can resume if interrupted
if os.path.isfile(OUTPUT_PATH):
    representations = torch.load(OUTPUT_PATH, weights_only=True)
    print(f"Resuming: {len(representations)} embeddings already saved")
else:
    representations = {}

remaining = [(acc, seq) for acc, seq in data if acc not in representations]
print(f"Remaining to process: {len(remaining)}")

for batch_start in range(0, len(remaining), BATCH_SIZE):
    batch = remaining[batch_start : batch_start + BATCH_SIZE]

    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[REPR_LAYER], return_contacts=False)

    token_repr = results["representations"][REPR_LAYER]  # (B, L+2, D)

    for i, (acc, _) in enumerate(batch):
        seq_len = batch_lens[i].item()
        # slice off BOS and EOS tokens → shape (seq_len, 1280)
        representations[acc] = token_repr[i, 1 : seq_len - 1].cpu()

    processed = batch_start + len(batch)
    print(f"  [{processed}/{len(remaining)}] last batch: {[a for a, _ in batch]}")

    # Save incrementally every batch so progress isn't lost on crash
    torch.save(representations, OUTPUT_PATH)

print(f"\nDone. {len(representations)} embeddings saved to {OUTPUT_PATH}")
print("Tensor shapes (first 3):")
for acc, t in list(representations.items())[:3]:
    print(f"  {acc}: {tuple(t.shape)}")