from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string
from pathlib import Path

import numpy as np
from Bio import SeqIO, Phylo
import pandas as pd
from scipy.spatial.distance import squareform, pdist, cdist
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
from time import time

import argparse
import pyximport
pyximport.install()

from aux_msa_functions import *
from MSAGeneratorESM import MSAGeneratorESM
from MSA_phylogeny_class import Creation_MSA_Generation_MSA1b_Cython

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tool", action="store", dest="tool",
                    help="tool used for suimulation of MSA")

parser.add_argument("-M", "--input_MSA", action="store", dest="input_MSA",
                    help="input protein family MSA")

parser.add_argument("-t", "--input_tree", action="store", dest="input_tree",
                    help="input protein family tree")

parser.add_argument("-O", "--output", action="store", dest="output",
                    help="location of simulated MSA", type=int, default=0
                )

parser.add_argument("-n", "--num_seqs", action="store", dest="num_seqs",
                    help="number of sequences to simulate", type=int
                )

parser.add_argument("-s", "--start_seq_index", action="store", dest="start_seq_index",
                    help="index in MSA of starting sequence of simulation", type=int, default=0
                )

parser.add_argument("-c", "--context_type", action="store", dest="context",
                    help="define the type of context to use when using MSA transformer for simulation")

parser.add_argument("-cs", "--context_size", action="store", dest="context_size",
                    help="write the batch size", type=int)

parser.add_argument("--chunked", action="store_true", dest="chunked",
                    help="write the batch size", type=int)

args = parser.parse_args()

tool = args.tool
MSA_path = args.input_MSA
tree_path = args.input_tree
num_seqs = args.num_seqs
context_type = args.context_type
context_size = args.context_size
chunked = args.chunked
starting_seq_index = args.starting_seq_index
output = args.output

all_seqs = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(MSA_path, "fasta")]

if num_seqs == None:
    num_seqs = len(all_seqs)

tree = Phylo.read(tree_path,"newick")

if tool == "MSA-1b":

    MSA_gen_obj = Creation_MSA_Generation_MSA1b_Cython(MSA = all_seqs, full_tree = tree, full_tree_path = tree_path)

    t1 = time()
    method = "minimal"
    masked = True
    chunked = True

    if not chunked:
        new_MSA = MSA_gen_obj.msa_tree_phylo(tree.clade,flip_before_start=0, method=method, masked=masked, context_type = "dynamic")
    else:
        new_MSA = MSA_gen_obj.msa_tree_phylo_chunked(total_sequences = 100,sequences_per_iteration = 10, method=method, masked=masked)
    t2 = time()

    Seq_tuples_to_fasta(new_MSA,output)

elif tool == "ESM2":

    ESM_gen_obj = MSAGeneratorESM(number_of_nodes= len(all_seqs[0][1]), number_state_spin=21,
                                  batch_size=1,model="facebook/esm2_t6_8M_UR50D")
    
    first_sequence = all_seqs[starting_seq_index][1]
    new_MSA = ESM_gen_obj.msa_tree_phylo(clade_root=tree.clade, first_sequence=first_sequence, flip_before_start=0)

    seq_records = []
    for i in range(new_MSA.shape[0]):
        seq_records.append(SeqRecord(seq=Seq(''.join(ESM_gen_obj.inverse_amino_acid_map[index]
                                                     for index in new_MSA[i])),
                                     id='seq' + str(i)))
    SeqIO.write(seq_records,f"{output}.fasta", "fasta")

elif tool == "Potts":

    ESM_gen_obj = MSAGeneratorESM(number_of_nodes= len(all_seqs[0][1]), number_state_spin=21,
                                  batch_size=1,model="facebook/esm2_t6_8M_UR50D")
    
    first_sequence = all_seqs[starting_seq_index][1]
    new_MSA = ESM_gen_obj.msa_tree_phylo(clade_root=tree.clade, first_sequence=first_sequence, flip_before_start=0)
    
    seq_records = []
    for i in range(new_MSA.shape[0]):
        seq_records.append(SeqRecord(seq=Seq(''.join(ESM_gen_obj.inverse_amino_acid_map[index]
                                                     for index in new_MSA[i])),
                                     id='seq' + str(i)))
    SeqIO.write(seq_records,f"{output}.fasta", "fasta")

print((t2-t1)/60)
