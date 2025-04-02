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

# if __name__ == "main":

parser = argparse.ArgumentParser()


parser.add_argument("-O", "--output", action="store", dest="output",
                    help="path to final output"
                )

parser.add_argument("-i", "--input_MSA", action="store", dest="input_MSA",
                    help="path to natural seed MSA"
                )

parser.add_argument("--pseudocount", action="store", dest="pseudocount", 
                    help="Method of calculating MI", type=float, default=0.0)

parser.add_argument("--proposal_type", action="store", dest="proposal_type",
                    help="proposal distribution used")

parser.add_argument("--sim_ind",action="store",dest="sim_ind",
                    help="simulation index", type=int)

args = parser.parse_args()

output = args.output
input_MSA_sequence = args.input_MSA
pseudocount = args.pseudocount
proposal_type = args.proposal_type
sim_ind = args.sim_ind


input_MSA_sequence = pd.read_csv(input_MSA_sequence, delimiter = "\t")
reference_MSA = input_MSA_sequence.loc[input_MSA_sequence["n_mutations"] == 0, :]
nat_array = pd.DataFrame([list(seq) for seq in reference_MSA["sequence"]])
n_mutations_list = list(input_MSA_sequence["n_mutations"].unique())

output_df = []

for n_mutations in n_mutations_list:

    if n_mutations == 0:
        continue

    current_MSA = input_MSA_sequence.loc[input_MSA_sequence["n_mutations"] == n_mutations, :]

    sim_array = pd.DataFrame([list(seq) for seq in current_MSA["sequence"]]) 

    mi_sim_values = []

    for k in range(sim_array.shape[1]):

        mi_sim = custom_mi(list(sim_array.iloc[:,k]),list(nat_array.iloc[:,k]), pseudocount=pseudocount)
        mi_sim_values.append(mi_sim)

    mi_sim_values_mean = np.average(mi_sim_values)

    output_df.append({"sim_ind":sim_ind,"proposal_type":proposal_type, "pseudocount":pseudocount, "n_mutations":n_mutations, "mean MI value": mi_sim_values_mean})

output_df = pd.DataFrame(output_df)

print(output)

output_df.to_csv(output, sep='\t', index=False)




        
