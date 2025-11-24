import re
import pandas as pd
import torch 
import os

from .sampling import BASES
from src.utils.tokenizer import ResidueTokenizer


def read_protein_from_csv(protein_name, file_path):
    try:
        data = pd.read_csv(file_path)
        if 'prot_name' not in data.columns or 'seq' not in data.columns:
            raise ValueError("The CSV file must contain 'prot_name' and 'seq' columns.")
        result = data.loc[data['prot_name'] == protein_name.lower(), 'seq']
        if not result.empty:
            return result.iloc[0]
        else:
            return None  # Return None if not found
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def clean_rna_sequence(rna_sequence: str) -> str:
    if rna_sequence.startswith('RNA:'):
        rna_sequence = rna_sequence[4:]
    clean_seq = rna_sequence.upper().strip()
    clean_seq = clean_seq.replace('T', 'U')
    clean_seq = re.sub(r'[^AUCGXN-]', '', clean_seq)
    return clean_seq


def save_to_fasta(sequences, filename="rna_sequences.fasta", headers=None):
    with open(filename, "w") as fasta_file:
        for i, seq in enumerate(sequences):
            if headers:
                fasta_file.write(f">{headers[i]}\n")
                fasta_file.write(f"{seq}\n")
            else:
                fasta_file.write(f">RNAtranslatorXraw_ _RNA{i}\n")
                fasta_file.write(f"{seq}\n")

def _rna_base_token_ids(tokenizer):
    tids = []
    for b in BASES:
        tid = tokenizer.convert_tokens_to_ids(f"RNA_{b}")
        if tid is None or tid < 0:
            raise ValueError(f"Missing tokenizer id for RNA_{b}")
        tids.append(tid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(tids, device=device, dtype=torch.long)

def read_fasta_sequences(path):
    seqs = []
    if not os.path.exists(path):
        return seqs
    header, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seqs.append((header, ''.join(seq)))
                header = line[1:].strip()
                seq = []
            else:
                seq.append(line)
        if header is not None:
            seqs.append((header, ''.join(seq)))
    return seqs