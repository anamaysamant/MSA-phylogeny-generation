import logging
import numpy as np
from Bio import SeqIO
import pandas as pd
import torch
import os
import argparse
import matplotlib.pyplot as plt
import esm

from aux_msa_functions import *
from sklearn.metrics import mutual_info_score
from select_gpu import get_free_gpu

work_dir = os.getcwd()
os.chdir("./scripts")
from MSA_phylogeny_class import Creation_MSA_Generation_MSA1b_Cython
os.chdir(work_dir)

import os
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu())


parser = argparse.ArgumentParser()


parser.add_argument("-O", "--output", action="store", dest="output",
                    help="path to final output"
                )

parser.add_argument("-i", "--input_MSA", action="store", dest="input_MSA",
                    help="path to natural seed MSA"
                )

parser.add_argument("-cs", "--context_size", action="store", dest="context_size",
                    help="size of context for MSA-1b simulation along phylogeny", type=int)

parser.add_argument( "--n_mutations_interval", action="store", dest="n_mutations_interval", 
                    help="number of mutations per round of evolution", type=int)

parser.add_argument( "--n_rounds", action="store", dest="n_rounds", 
                    help="number of rounds of evolution", type=int)

parser.add_argument( "--n_sequences", action="store", dest="n_sequences", 
                    help="number of sequences to analyze", type=int)

parser.add_argument( "--random", action="store_true", dest="random", 
                    help="start with a random MSA")

parser.add_argument( "--FT_fam", action="store", dest="FT_fam", 
                    help="family on which MSA transformer is finetuned")


parser.add_argument("--pseudocount", action="store", dest="pseudocount", 
                    help="Method of calculating MI", type=float, default=0.0)

parser.add_argument( "--seed", action="store", dest="seed", 
                    help="random seed to use", type=int, default=0)

parser.add_argument("--proposal_type", action="store", dest="proposal_type",
                    help="proposal distribution used")

parser.add_argument("-s", "--start_seqs", action="store", dest="start_seqs",
                    help="index in MSA of starting sequence of simulation", default="sampled")

args = parser.parse_args()

context_size = args.context_size
output = args.output
n_mutations_interval = args.n_mutations_interval
input_MSA = args.input_MSA
n_sequences = args.n_sequences
n_rounds = args.n_rounds
FT_fam = args.FT_fam
random = args.random
pseudocount = args.pseudocount
proposal_type = args.proposal_type
seed = args.seed
start_seqs = args.start_seqs

np.random.seed(seed)

# logging.basicConfig(
#     filename=f"MI_decay_{context_size}_nseqs_{n_sequences}_pseudo_{pseudocount}.log",               
#     level=logging.INFO,              
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filemode="a"
# )

all_seqs = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(input_MSA, "fasta")]

model_to_use, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()

if FT_fam != None:
    model_to_use = torch.load(f"./finetuned_MSA_models/MSA_finetuned_{FT_fam}.pt")


method = "minimal"
masked = True

output_df = []

if start_seqs == "random":
    char_list = list("-ACDEFGHIKLMNPQRSTVWY")
    sampled_seqs = []
    for i in range(n_sequences):
        rand_char_order = np.random.choice(range(len(char_list)), len(all_seqs[0][1]), replace = True)
        rand_seq = [char_list[i] for i in rand_char_order]
        rand_seq = ''.join(rand_seq)
        sampled_seqs.append((f'seq{i}',rand_seq))
else:
    sampled_seq_inds = np.random.choice(range(len(all_seqs)), n_sequences, replace=False)
    sampled_seqs = [all_seqs[i] for i in sampled_seq_inds]

n_mutations = 0
old_MSA = sampled_seqs.copy()

for i in range(len(sampled_seqs)):

    output_df.append({"MSA_id":0,"n_mutations":0, "sequence_name":sampled_seqs[i][0],"sequence":sampled_seqs[i][1]})

for j in range(1, n_rounds + 1):


    t1 = time()

    # logging.info(f"Simulating round {j} of {proposal_type} proposal")

    new_MSA = []
    n_mutations += n_mutations_interval

    for i in range(len(old_MSA)):

        seq_name = old_MSA[i][0]
        all_seqs[0] = old_MSA[i]
        MSA_gen_obj = Creation_MSA_Generation_MSA1b_Cython(MSA = all_seqs, start_seq_index=0, model_to_use=model_to_use, alphabet = alphabet, seed=seed)

        new_MSA_seq = MSA_gen_obj.msa_no_phylo(context_size = context_size, n_sequences = 1,n_mutations = n_mutations_interval, method=method, 
                                            masked=masked, proposal = proposal_type)
        
        new_MSA.append((seq_name,new_MSA_seq[0][1]))

        output_df.append({"MSA_id":j,"n_mutations":n_mutations, "sequence_name":seq_name,"sequence":new_MSA_seq[0][1]})


    old_MSA = new_MSA.copy()
    del new_MSA

    t2 = time()

    # logging.info(f"Finished round {j} of {proposal_type} proposal. Time taken: {(t2 -t1)/60} minutes")

output_df = pd.DataFrame(output_df)

output_df.to_csv(output, sep='\t', index=False)

