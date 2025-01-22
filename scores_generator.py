import argparse
import pandas as pd
from aux_msa_functions import *
import time
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_hmmer", action="store", dest="input_hmmer",
                    help="unprocessed hmmer table")

parser.add_argument("--J_params", action="store", dest="J_params",
                    help="bmDCA J params")

parser.add_argument("--h_params", action="store", dest="h_params",
                    help="bmDCA h parameters")

parser.add_argument("-M", "--simulated_MSA", action="store", dest="simulated_MSA",
                    help="MSA resulting from simulation")

parser.add_argument("--original_MSA", action="store", dest="original_MSA",
                    help="original MSA used for simulation")

parser.add_argument("-O", "--output", action="store", dest="output", 
                    help="processed hmmer table")

args = parser.parse_args()

input_hmmer = args.input_hmmer
output = args.output
simulated_MSA = args.simulated_MSA
original_MSA = args.original_MSA
J_params = args.J_params
h_params = args.h_params

table = open(input_hmmer)
with open(output,"w") as f:
    line = table.readline()
    while line: 
        if not line.startswith("#"):
            
            f.writelines(line)
            
        line = table.readline()

relevant_cols = ["sequence_name","hmmer_seq_score"]
scores_table = pd.read_csv(output, delimiter="\s+",header=None, usecols=[0,5], names=relevant_cols)

all_seqs = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(simulated_MSA, "fasta")]
all_seqs_nat = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(original_MSA, "fasta")]

n_cols = len(all_seqs[0][1])
n_rows = len(all_seqs)

all_seqs = pd.DataFrame(all_seqs, columns=["sequence_name","sequence"])

scores_table = scores_table.merge(all_seqs, on="sequence_name")
scores_table = scores_table[["sequence_name","sequence","hmmer_seq_score"]]

bmdca_mapping  = {k:v for k,v in zip(list("-ACDEFGHIKLMNPQRSTVWY"), range(21))}

J_params = np.load(J_params)
h_params = np.load(h_params)

stat_energy_scores = []
for sequence in scores_table["sequence"]:
    hamiltonian = 0
    num_sequence = [bmdca_mapping[char] for char in list(sequence)]
    for node_i in range(n_cols):
        hamiltonian -= h_params[node_i,num_sequence[node_i]]
        for index_neighboor in range(node_i+1,n_cols):
            hamiltonian -= J_params[node_i,index_neighboor,num_sequence[node_i],num_sequence[index_neighboor]]

    stat_energy_scores.append(-hamiltonian)

scores_table["stat_energy_scores"] = stat_energy_scores

sim_sequences_array = np.array([list(seq) for seq in scores_table["sequence"]], dtype=np.bytes_).view(np.uint8)
nat_sequences_array = np.array([list(seq) for _,seq in all_seqs_nat], dtype=np.bytes_).view(np.uint8)

distance_matrix = cdist(sim_sequences_array, nat_sequences_array, "hamming")
min_natural_ham_distance = list(distance_matrix.min(axis = 1))
max_natural_ham_distance = list(distance_matrix.max(axis = 1))

scores_table["min_natural_ham_dist"] = min_natural_ham_distance
scores_table["max_natural_ham_dist"] = max_natural_ham_distance

scores_table.to_csv(output, sep="\t", index = False)



    




