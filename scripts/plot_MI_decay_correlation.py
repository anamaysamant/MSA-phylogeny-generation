import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
from aux_msa_functions import *


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

np.random.seed(42)

MI_files = snakemake.input["MI_files"] # type: ignore
seed_MSA = snakemake.input["seed_MSA"] # type: ignore
output = snakemake.output[0] # type: ignore

seed_MSA = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(seed_MSA, "fasta")]

full_dataframe = pd.read_csv(MI_files[0], delimiter = "\t")

n_mutations_list = list(full_dataframe["n_mutations"])

for i in range(1, len(MI_files)):

    cur_dataframe = pd.read_csv(MI_files[i], delimiter = "\t")
    full_dataframe = pd.concat([full_dataframe, cur_dataframe]).reset_index(drop=True)

n_sequences = int(snakemake.wildcards["n_sequences_MI"]) # type: ignore
pseudocount = float(snakemake.wildcards["pseudocount"]) # type: ignore

nat_df = pd.DataFrame()

for i in range(10):

    cur_df = pd.DataFrame()

    cur_df["n_mutations"] = n_mutations_list

    random_ind_1 = list(np.random.choice(range(1,len(seed_MSA)),n_sequences, replace = False))
    rand_MSA_1 = [seed_MSA[i] for i in random_ind_1]

    remaining_inds = list(set(range(len(seed_MSA))).difference(set(random_ind_1)))

    random_ind_2 = list(np.random.choice(remaining_inds,n_sequences, replace = False))
    rand_MSA_2 = [seed_MSA[i] for i in random_ind_2]

    sim_array_1 = pd.DataFrame([list(seq[1]) for seq in rand_MSA_1]) 
    sim_array_2 = pd.DataFrame([list(seq[1]) for seq in rand_MSA_2]) 

    mi_sim_values = []

    for k in range(sim_array_1.shape[1]):

        mi_sim = custom_mi(list(sim_array_1.iloc[:,k]),list(sim_array_2.iloc[:,k]), pseudocount=pseudocount)
        mi_sim_values.append(mi_sim)

    mi_sim_values_mean = np.average(mi_sim_values)

    cur_df["sim_ind"] = i + 1
    cur_df["mean MI value"] = mi_sim_values_mean

    nat_df = pd.concat([nat_df, cur_df]).reset_index(drop=True)


fig, axes = plt.subplots(ncols=1, nrows=1)

sns.lineplot(data = full_dataframe, x="n_mutations", y = "mean MI value", hue ="proposal_type" ,ax=axes)
sns.lineplot(data = nat_df, x="n_mutations", y = "mean MI value",ax=axes, label = "natural MI")
axes.annotate(f'avg_nat_MI = {np.mean(nat_df["mean MI value"]):.2f}',xy = (0.3,0.9), xycoords = "axes fraction", fontsize = 15)

plt.legend()

plt.savefig(output)


