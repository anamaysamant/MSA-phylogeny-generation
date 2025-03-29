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
from time import time, sleep


import subprocess

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

            mi += joint_probs * np.log(joint_probs/(probs_elem_1 * probs_elem_2))


    return mi





parser = argparse.ArgumentParser()


parser.add_argument("-O", "--output", action="store", dest="output",
                    help="path to final output"
                )

parser.add_argument("-i", "--input_MSA", action="store", dest="input_MSA",
                    help="path to natural seed MSA"
                )

parser.add_argument("--path_to_HMM", action="store", dest="path_to_HMM",
                    help="path to HMM")

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
path_to_HMM = args.path_to_HMM

proposal_distributions = ["msa_prob_dist","random"]
seed = args.seed

np.random.seed(seed)

logging.basicConfig(
    filename=f"hmmer_convergence_{context_size}_nseqs_{n_sequences}.log",               
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


model_to_use, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()

if FT_fam != None:
    model_to_use = torch.load(f"./finetuned_MSA_models/MSA_finetuned_{FT_fam}.pt")

method = "minimal"
masked = True

output_df = []

nat_array = pd.DataFrame([list(seq[1]) for seq in sampled_seqs])
mi_nat_values = []

old_MSA = sampled_seqs.copy()

final_hmmer_table = pd.DataFrame(columns=["sequence_name","n_mutations","proposal_type","hmmer_seq_score"])

Seq_tuples_to_fasta(old_MSA, file_path="tmp_MSA.fasta")

with open('tmp_MSA_ungapped.fasta',"w+") as f:
    subprocess.run(['seqkit', 'replace','-s','-p','-','-r','','tmp_MSA.fasta'], stdout = f) 


subprocess.run(['hmmsearch','--max','--tblout','hmm_table.tbl',path_to_HMM,'tmp_MSA_ungapped.fasta'])

subprocess.run(['rm','tmp_MSA.fasta'])
subprocess.run(['rm','tmp_MSA_ungapped.fasta'])

table = open("hmm_table.tbl")
with open("hmm_table_processed.tsv","w") as f:
    line = table.readline()
    while line: 
        if not line.startswith("#"):
            
            f.writelines(line)
            
        line = table.readline()

subprocess.run(['rm','hmm_table.tbl'])

for proposal_type in proposal_distributions:

    relevant_cols = ["sequence_name","hmmer_seq_score"]
    scores_table = pd.read_csv("hmm_table_processed.tsv", delimiter="\s+",header=None, usecols=[0,5], names=relevant_cols)
    if len(scores_table) > 0:
        scores_table["n_mutations"] = 0
        scores_table["proposal_type"] = proposal_type

        final_hmmer_table = pd.concat([final_hmmer_table, scores_table]).reset_index(drop=True)

subprocess.run(['rm','hmm_table_processed.tsv'])

for proposal_type in proposal_distributions:

    n_mutations = 0
    old_MSA = sampled_seqs.copy()

    for j in range(n_rounds):

        new_MSA = []

        logging.info(f"Simulating round {j} of {proposal_type} proposal")

        t1 = time()

        for i in range(len(old_MSA)):

            all_seqs[0] = old_MSA[i]
            MSA_gen_obj = Creation_MSA_Generation_MSA1b_Cython(MSA = all_seqs, start_seq_index=0, model_to_use=model_to_use, seed=seed)

            new_MSA_seq = MSA_gen_obj.msa_no_phylo(context_size = context_size, n_sequences = 1,n_mutations = n_mutations_interval, method=method, 
                                                masked=masked, proposal = proposal_type)
            
            
            new_MSA.append((f"seq{i}",new_MSA_seq[0][1]))

        n_mutations += n_mutations_interval

        Seq_tuples_to_fasta(new_MSA,"tmp_MSA.fasta")

        with open('tmp_MSA_ungapped.fasta',"w+") as f:
            subprocess.run(['seqkit', 'replace','-s','-p','-','-r','','tmp_MSA.fasta'], stdout = f)  
        subprocess.run(['hmmsearch','--max','--tblout','hmm_table.tbl',path_to_HMM,'tmp_MSA_ungapped.fasta'])

        subprocess.run(['rm','tmp_MSA.fasta'])
        subprocess.run(['rm','tmp_MSA_ungapped.fasta'])

        table = open("hmm_table.tbl")
        with open("hmm_table_processed.tsv","w") as f:
            line = table.readline()
            while line: 
                if not line.startswith("#"):
                    
                    f.writelines(line)
                    
                line = table.readline()

        subprocess.run(['rm','hmm_table.tbl'])

        relevant_cols = ["sequence_name","hmmer_seq_score"]
        scores_table = pd.read_csv("hmm_table_processed.tsv", delimiter="\s+",header=None, usecols=[0,5], names=relevant_cols)

        if len(scores_table) > 0:
            scores_table["n_mutations"] = n_mutations
            scores_table["proposal_type"] = proposal_type

            scored_seqs = list(scores_table["sequence_name"])
            all_possible_seqs = [f"seq{i}" for i in range(n_sequences)]

            add_df_entries = []

            for seq in all_possible_seqs:
                if seq not in scored_seqs:
                    add_df_entries.append({"sequence_name":seq, "n_mutations":n_mutations, "proposal_type":proposal_type, "hmmer_seq_score": 0})

            if len(add_df_entries) > 0:
                add_df_entries = pd.DataFrame(add_df_entries)
                scores_table = pd.concat([scores_table, add_df_entries]).reset_index(drop=True)

            final_hmmer_table = pd.concat([final_hmmer_table, scores_table]).reset_index(drop=True)

        subprocess.run(['rm','hmm_table_processed.tsv'])

        old_MSA = new_MSA.copy()
        del new_MSA

        t2 = time()

        logging.info(f"Finished round {j} of {proposal_type} proposal. Time taken: {(t2 -t1)/60} minutes")
    
    Seq_tuples_to_fasta(old_MSA,f'MSA_hmmer_convergence_{seed}_{n_rounds*n_mutations_interval}_muts_{proposal_type}prop.fasta')

        


final_hmmer_table.to_csv(f'hmmer_convergence_seed_{seed}_{n_rounds*n_mutations_interval}_muts.tsv', sep='\t', index=False)

import seaborn as sns

fig, axes = plt.subplots(ncols=1, nrows=1)

sns.lineplot(data = final_hmmer_table, x="n_mutations", y = "hmmer_seq_score", hue ="proposal_type", units="sequence_name", estimator=None ,ax=axes)
plt.legend()

plt.savefig(f'hmmer_convergence_seed_{seed}_{n_rounds*n_mutations_interval}_muts.png')

fig, axes = plt.subplots(ncols=1, nrows=1)

sns.lineplot(data = final_hmmer_table, x="n_mutations", y = "hmmer_seq_score", hue ="proposal_type",ax=axes)
plt.legend()

plt.savefig(f'hmmer_convergence_seed_{seed}_{n_rounds*n_mutations_interval}_muts_avg.png')


