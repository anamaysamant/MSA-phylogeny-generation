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

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu())

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

            if probs_elem_1 > 0 and probs_elem_2 > 0:
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

parser.add_argument( "--MI_method", action="store", dest="MI_method", 
                    help="Method of calculating MI")

parser.add_argument("--pseudocount", action="store", dest="pseudocount", 
                    help="Method of calculating MI", type=float, default=0.0)

parser.add_argument( "--seed", action="store", dest="seed", 
                    help="random seed to use", type=int, default=0)


args = parser.parse_args()

context_size = args.context_size
output = args.output
n_mutations_interval = args.n_mutations_interval
input_MSA = args.input_MSA
n_sequences = args.n_sequences
n_rounds = args.n_rounds
FT_fam = args.FT_fam
random = args.random
MI_method = args.MI_method
pseudocount = args.pseudocount

proposal_distributions = ["msa_prob_dist","random"]
seed = args.seed

np.random.seed(seed)

logging.basicConfig(
    filename=f"MI_decay_{context_size}_nseqs_{n_sequences}_pseudo_{pseudocount}.log",               
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode="a"
)

all_seqs = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(input_MSA, "fasta")]

if random:
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

if FT_fam != None:
    model_to_use = torch.load(f"./finetuned_MSA_models/MSA_finetuned_{FT_fam}.pt")

else:
    model_to_use = None

method = "minimal"
masked = True

output_df = []

nat_array = pd.DataFrame([list(seq[1]) for seq in sampled_seqs])
mi_nat_values = []

logging.info(f"Using {MI_method} for MI decay analysis")

if MI_method == "within_MSA":

    for k in range(nat_array.shape[1]):

        for l in range(k + 1, nat_array.shape[1]):

            mi_nat = custom_mi(list(nat_array.iloc[:,k]),list(nat_array.iloc[:,l]), pseudocount=pseudocount)
            mi_nat_values.append(mi_nat)

    mi_nat_values_mean = np.average(mi_nat_values)

    old_MSA = sampled_seqs.copy()

    for proposal_type in proposal_distributions:

        n_mutations = 0

        output_df.append({"proposal_type":proposal_type, "n_mutations":n_mutations, "mean MI value": mi_nat_values_mean})
        
        for i in range(n_rounds):

            new_MSA = []

            logging.info(f"Simulating round {i} of {proposal_type} proposal")

            t1 = time()

            for i in range(len(old_MSA)):

                np.random.seed(seed)

                all_seqs[0] = old_MSA[i]
                MSA_gen_obj = Creation_MSA_Generation_MSA1b_Cython(MSA = all_seqs, start_seq_index=0, model_to_use=model_to_use)

                new_MSA_seq = MSA_gen_obj.msa_no_phylo(context_size = context_size, n_sequences = 1,n_mutations = n_mutations_interval, method=method, 
                                                    masked=masked, proposal = proposal_type)
                
                
                new_MSA.append((f"seq{i}",new_MSA_seq[0][1]))

            sim_array = pd.DataFrame([list(seq[1]) for seq in new_MSA]) 

            mi_sim_values = []

            for k in range(sim_array.shape[1]):

                for l in range(k + 1, sim_array.shape[1]):

                    mi_sim = custom_mi(list(sim_array.iloc[:,k]),list(sim_array.iloc[:,l]), pseudocount=pseudocount)

                    mi_sim_values.append(mi_sim)

            mi_sim_values_mean = np.average(mi_sim_values)
            n_mutations += n_mutations_interval

            output_df.append({"proposal_type":proposal_type, "n_mutations":n_mutations, "mean MI value": mi_sim_values_mean})

            old_MSA = new_MSA.copy()
            del new_MSA

            t2 = time()

            logging.info(f"Finished round {i} of {proposal_type} proposal. Time taken: {(t2 -t1)/60} minutes")

    output_df = pd.DataFrame(output_df)

    output_df.to_csv('MI_decay_df_within_MSA.tsv', sep='\t', index=False)

elif MI_method == "corresponding_columns":

    old_MSA = sampled_seqs.copy()

    for proposal_type in proposal_distributions:

        n_mutations = 0
        
        for i in range(n_rounds):

            new_MSA = []

            t1 = time()

            logging.info(f"Simulating round {i} of {proposal_type} proposal")

            for i in range(len(old_MSA)):

                np.random.seed(seed)

                all_seqs[0] = old_MSA[i]
                MSA_gen_obj = Creation_MSA_Generation_MSA1b_Cython(MSA = all_seqs, start_seq_index=0, model_to_use=model_to_use)

                new_MSA_seq = MSA_gen_obj.msa_no_phylo(context_size = context_size, n_sequences = 1,n_mutations = n_mutations_interval, method=method, 
                                                    masked=masked, proposal = proposal_type)
                
                new_MSA.append((f"seq{i}",new_MSA_seq[0][1]))

            sim_array = pd.DataFrame([list(seq[1]) for seq in new_MSA]) 

            mi_sim_values = []

            for k in range(sim_array.shape[1]):

                mi_sim = custom_mi(list(sim_array.iloc[:,k]),list(nat_array.iloc[:,k]), pseudocount=pseudocount)
                mi_sim_values.append(mi_sim)

            mi_sim_values_mean = np.average(mi_sim_values)
            n_mutations += n_mutations_interval

            output_df.append({"proposal_type":proposal_type, "n_mutations":n_mutations, "mean MI value": mi_sim_values_mean})

            old_MSA = new_MSA.copy()
            del new_MSA

            t2 = time()

            logging.info(f"Finished round {i} of {proposal_type} proposal. Time taken: {(t2 -t1)/60} minutes")

    output_df = pd.DataFrame(output_df)

    output_df.to_csv('MI_decay_df_corr_cols.tsv', sep='\t', index=False)


import seaborn as sns

fig, axes = plt.subplots(ncols=1, nrows=1)

sns.lineplot(data = output_df, x="n_mutations", y = "mean MI value", hue ="proposal_type" ,ax=axes)
plt.legend()

plt.savefig(f'MI_decay_{MI_method}_pseudo_{pseudocount}_{n_sequences}_seqs.png')