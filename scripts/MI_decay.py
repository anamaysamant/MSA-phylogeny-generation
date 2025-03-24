import logging
import numpy as np
from Bio import SeqIO
import pandas as pd
import torch
import os
import argparse
import matplotlib.pyplot as plt

from aux_msa_functions import *
from sklearn.metrics import mutual_info_score
from select_gpu import get_free_gpu

work_dir = os.getcwd()
os.chdir("./scripts")
from MSA_phylogeny_class import Creation_MSA_Generation_MSA1b_Cython
os.chdir(work_dir)

import os
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def custom_mi(values_1, values_2, domain_1 = list("-ACDEFGHIKLMNPQRSTVWY"), domain_2 = list("-ACDEFGHIKLMNPQRSTVWY"), pseudocount = 0):

    assert len(values_1) == len(values_2), 'arrays must be of the same length'
    
    values_1_counts = {}
    values_2_counts = {}
    joint_counts = {}
    mi = 0

    for elem_1 in domain_1:
        for elem_2 in domain_2:
            joint_counts[(elem_1, elem_2)] = pseudocount
    
    for elem_1 in domain_1:
        values_1_counts[elem_1] = pseudocount * len(domain_2)
    
    for elem_2 in domain_2:
        values_2_counts[elem_2] = pseudocount * len(domain_1)

    for i in range(len(values_1)):

        elem_1 = values_1[i]
        elem_2 = values_2[i]

        values_1_counts[elem_1] += 1
        values_2_counts[elem_2] += 1

        joint_counts[(elem_1, elem_2)] += 1

    total_counts = pseudocount * len(domain_1) * len(domain_2) + len(values_1)
    
    for elem_1 in domain_1:
        for elem_2 in domain_2:

            probs_elem_1 = values_1_counts[elem_1] / total_counts
            probs_elem_2 = values_2_counts[elem_2] / total_counts
            joint_probs = joint_counts[(elem_1, elem_2)] / total_counts

            if joint_probs > 0:
                mi += joint_probs * np.log(joint_probs/(probs_elem_1 * probs_elem_2))


    return mi





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
                    help="index in MSA of starting sequence of simulation", type=int, default="sampled")

args = parser.parse_args()

output = args.output
input_MSA_sequence = args.input_MSA_sequence
n_sequences = args.n_sequences
MI_method = args.MI_method
pseudocount = args.pseudocount
proposal_type = args.proposal_type

# if MI_method == "corresponding_columns":


reference_MSA = input_MSA_sequence.loc[input_MSA_sequence["n_mutations"] == 0, :]
nat_array = pd.DataFrame([list(seq) for seq in reference_MSA["sequence"]])
n_mutations_list = list(input_MSA_sequence["n_mutations"].unique())
sim_ind = input_MSA_sequence.split("-")[-1]

output_df = []

for n_mutations in n_mutations_list:

    if n_mutations == 0:
        continue

    current_MSA = input_MSA_sequence[input_MSA_sequence["n_mutations"] == n_mutations_list, :]

    sim_array = pd.DataFrame([list(seq) for seq in current_MSA]) 

    mi_sim_values = []

    for k in range(sim_array.shape[1]):

        mi_sim = custom_mi(list(sim_array.iloc[:,k]),list(nat_array.iloc[:,k]), pseudocount=pseudocount)
        mi_sim_values.append(mi_sim)

    mi_sim_values_mean = np.average(mi_sim_values)

    output_df.append({"proposal_type":proposal_type, "n_mutations":n_mutations, "mean MI value": mi_sim_values_mean, "sim_ind":sim_ind})

# import seaborn as sns

# fig, axes = plt.subplots(ncols=1, nrows=1)

# sns.lineplot(data = output_df, x="n_mutations", y = "mean MI value", hue ="proposal_type" ,ax=axes)
# plt.legend()

# plt.savefig(f'MI_decay_{MI_method}_pseudo_{pseudocount}_{n_sequences}_seqs_interval_{n_mutations_interval}.png')


        
