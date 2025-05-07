from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string
from pathlib import Path
import sys

import numpy as np
from Bio import SeqIO, Phylo
import pandas as pd
from scipy.spatial.distance import squareform, pdist, cdist
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
from time import time
import os

import argparse

from aux_msa_functions import *
from MSAGeneratorESM import MSAGeneratorESM
from MSAGeneratorPottsModel import MSAGeneratorPottsModel
from select_gpu import get_free_gpu

work_dir = os.getcwd()
os.chdir("./scripts")
from MSA_phylogeny_class import Creation_MSA_Generation_MSA1b_Cython
os.chdir(work_dir)

import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu()) 

starting_point_simulations = 500

ending_point_simulations = 1000

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--tool", action="store", dest="tool",
                    help="tool used for simulation of MSA")

parser.add_argument("-O", "--output", action="store", dest="output",
                    help="path to final simulated MSA"
                )

parser.add_argument("-i", "--input_MSA", action="store", dest="input_MSA",
                    help="path to natural seed MSA"
                )

parser.add_argument("-s", "--simulated_MSA", action="store", dest="simulated_MSA",
                    help="path to starting simulated MSA"
                )

parser.add_argument("--J_params", action="store", dest="J_params",
                    help="bmDCA J params")

parser.add_argument("--h_params", action="store", dest="h_params",
                    help="bmDCA h parameters")

parser.add_argument("-cs", "--context_size", action="store", dest="context_size",
                    help="size of context for MSA-1b simulation along phylogeny", type=int)

parser.add_argument("--proposal_type", action="store", dest="proposal_type",
                    help="proposal distribution used")


parser.add_argument( "--n_mutations_end", action="store", dest="n_mutations_end", 
                    help="number of mutations for independent MCMC sequence generation", type=int)

parser.add_argument( "--n_mutations_start", action="store", dest="n_mutations_start", 
                    help="starting point for independent MCMC sequence generation", type=int)

parser.add_argument( "--FT_fam", action="store", dest="FT_fam", 
                    help="family on which MSA transformer is finetuned")

parser.add_argument( "--seed", action="store", dest="seed", 
                    help="random seed to use", type=int, default=0)

args = parser.parse_args()

tool = args.tool
context_size = args.context_size
proposal_type = args.proposal_type
output = args.output
J_params = args.J_params
h_params = args.h_params
n_mutations_start = args.n_mutations_start
n_mutations_end = args.n_mutations_end
input_MSA = args.input_MSA
simulated_MSA = args.simulated_MSA

seed = args.seed
FT_fam = args.FT_fam

np.random.seed(seed)

all_seqs = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(input_MSA, "fasta")]
simulated_seqs = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(simulated_MSA, "fasta")]

add_mutations = n_mutations_end - n_mutations_start

assert add_mutations > 0

if tool == "MSA_1b":

    t1 = time()

    if FT_fam != None:
        model_to_use = torch.load(f"./finetuned_MSA_models/MSA_finetuned_{FT_fam}.pt")

    else:
        model_to_use = None

    method = "minimal"
    masked = True

    new_MSA = []

    for i in range(len(simulated_seqs)):

        all_seqs[0] = simulated_seqs[i]
        MSA_gen_obj = Creation_MSA_Generation_MSA1b_Cython(MSA = all_seqs, start_seq_index=0, model_to_use=model_to_use)

        new_MSA_seq = MSA_gen_obj.msa_no_phylo(context_size = context_size, n_sequences = 1,n_mutations = add_mutations, method=method, 
                                            masked=masked, proposal = proposal_type)
        
        new_MSA.append((f"seq{i}",new_MSA_seq[0][1]))

    t2 = time()

    Seq_tuples_to_fasta(new_MSA,output)
        
elif tool == "Potts":

    t1 = time()

    J_params = np.load(J_params)
    h_params = np.load(h_params)

    Potts_gen_obj = MSAGeneratorPottsModel(field = h_params, coupling = J_params)

    new_MSA = []

    for i in len(simulated_seqs):

        first_sequence = simulated_seqs[i]
        first_sequence = np.array([Potts_gen_obj.bmdca_mapping[char] for char in list(first_sequence)])

        new_MSA_seq = Potts_gen_obj.msa_no_phylo(n_sequences=1, n_mutations=add_mutations, first_sequence=first_sequence)

        new_MSA.append(new_MSA_seq)

    new_MSA = np.stack(new_MSA, axis=0)
    
    seq_records = []
    for i in range(new_MSA.shape[0]):
        seq_records.append(SeqRecord(seq=Seq(''.join(Potts_gen_obj.bmdca_mapping_inv[index]
                                                    for index in new_MSA[i])),
                                    id='seq' + str(i), description='seq' + str(i), name='seq' + str(i)))
    SeqIO.write(seq_records,output, "fasta")

    t2 = time() 


    