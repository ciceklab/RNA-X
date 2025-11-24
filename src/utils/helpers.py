import yaml
import os
import torch
import random
import numpy as np
import json
import pandas as pd
from collections import defaultdict
import pickle
from pathlib import Path
from typing import Dict, Any

# from generate import generate_
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform


def set_hyps(path, args):
    with open(path, errors="ignore") as f:
        hyps = yaml.safe_load(f)
        for k, v in hyps.items():
            setattr(args, k, v)
    return args

def reproducibility(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def get_random_rna(length):
    rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "T":3}

    rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

    mapping = dict(zip([0,1,2,3],"ACGU"))
    rsample = ''
    for i in range(length):
        p = random.random()
        if p < 0.3:
            rsample += 'C'
        elif p < 0.6:
            rsample += 'G'
        elif p < 0.8:
            rsample += 'A'
        else:
            rsample += 'U'

    return rsample


def shuffle_rna_sequences(rna_sequences):
    shuffled_sequences = []
    for rna in rna_sequences:
        rna_list = list(rna) 
        random.shuffle(rna_list)
        shuffled_sequences.append(''.join(rna_list))
    return shuffled_sequences

def read_rna_from_fasta(fasta_file_path):
    natural_rnas = []
    with open(fasta_file_path, "r") as file:
        sequence = ""
        for line in file:
            if line.startswith(">"):  
                if sequence:
                    natural_rnas.append(sequence.strip()) 
                    sequence = ""  
            else:
                sequence += line.strip()  
        if sequence:
            natural_rnas.append(sequence.strip())
    return natural_rnas

def read_rna_from_text(text_file_path):
    rna_sequences = []
    with open(text_file_path, "r") as file:
        for line in file:
            rna_sequences.append(line.strip()) 
    return rna_sequences

def read_protein_from_csv(protein_name, file_path):
    try:
        # print(protein_name)
        data = pd.read_csv(file_path)
        if 'prot_name' not in data.columns or 'seq' not in data.columns:
            raise ValueError("The CSV file must contain 'prot_name' and 'seq' columns.")
        result = data.loc[data['prot_name'] == protein_name, 'seq']
        # print(result)
    
        if not result.empty:
            return result.iloc[0]
        else:
            return f"Protein '{protein_name}' not found in the file."

    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def write_fasta(filename, rnas, prefix):
    with open(filename, "w") as fasta_file:
        for i, rna in enumerate(rnas, start=1):
            fasta_file.write(f">{prefix} RNA {i}\n{rna}\n")


def combine_fasta_files(output_filename, input_filenames):
    with open(output_filename, "w") as outfile:
        for input_file in input_filenames:
            with open(input_file, "r") as infile:
                outfile.write(infile.read())

def write_rna_to_fasta(fasta_file, title, sequence):
    fasta_file.write(f">{title}\n{sequence}\n")


def fasta_to_dict(fasta_file):
    rna_groups = {}
    current_group = None

    with open(fasta_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:]
                prefix = header.split('_')[0] + " " +header.split('_')[1]
                current_group = prefix
                if current_group not in rna_groups:
                    rna_groups[current_group] = []
            else:
                if current_group is not None:
                    rna_groups[current_group].append(line)
    return rna_groups

def calculate_kmer_features(rna_sequences, k):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), binary=True)
    return vectorizer.fit_transform(rna_sequences).toarray()

def tanimoto_similarity(matrix):
    def tanimoto(u, v):
        intersection = (u & v).sum()
        union = u.sum() + v.sum() - intersection
        return intersection / union if union > 0 else 0

    pairwise = pdist(matrix, metric=lambda u, v: tanimoto(u > 0, v > 0))
    return squareform(1 - pairwise)

def read_deepclip_output(json_path):
    with open(json_path, 'r') as f:
        json_file = f.read()
        data = json.loads(json_file)
        scores = {}

        for prediction in data.get("predictions", []):
            group_name = prediction["id"].split("_")[0]+ " "  + prediction["id"].split("_")[1]
            if group_name not in scores:
                scores[group_name] = []
            scores[group_name].append(prediction["score"])
    
    return scores

def read_protein_from_fasta(fasta_path):
    header = None
    sequence = ""
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:  # Already found a protein
                    break
                header = line[1:]  # skip '>'
            else:
                sequence += line.strip().replace("T", "U").upper()
    if not header or not sequence:
        raise ValueError(f"No protein found in {fasta_path}")
    return header, sequence

def parse_cluster_file(clstr_path):
    id2size = {}
    with open(clstr_path) as f:
        current = []
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                if current:
                    sz = len(current)
                    for seq_id in current:
                        id2size[seq_id] = sz
                current = []
            else:
                if ">" in line and "..." in line:
                    seq_id = line.split(">",1)[1].split("...",1)[0]
                    current.append(seq_id)
        if current:
            sz = len(current)
            for seq_id in current:
                id2size[seq_id] = sz
    return id2size

def parse_fasta(fa_path):
    seqs = {}
    with open(fa_path) as f:
        curr_id = None
        buf = []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if curr_id is not None:
                    seqs[curr_id] = "".join(buf)
                curr_id = line[1:]
                buf = []
            else:
                buf.append(line)
        if curr_id is not None:
            seqs[curr_id] = "".join(buf)
    return seqs


def load_pickle_dict(path: str) -> Dict[str, Any]:
    print(f"[load] {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data