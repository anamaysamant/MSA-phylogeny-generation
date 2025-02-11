import argparse
import pandas as pd
from aux_msa_functions import *
import time
from scipy.spatial.distance import cdist
from Bio import Phylo

def leaf_matcher(clade_root, all_syn_seqs, all_nat_seqs_dict):

    output = []
    
    def leaf_matcher_recur(tree_root, all_syn_seqs, all_nat_seqs_dict):
    
        b = tree_root.clades
        
        if len(b)>0:
            for clade in b:
               leaf_matcher_recur(clade, all_syn_seqs, all_nat_seqs_dict) 
        else:
            counter = len(output)
            output.append({"sequence_name":all_syn_seqs[counter][0],"corr_nat_seq_name":tree_root.name, "corr_nat_seq":all_nat_seqs_dict[tree_root.name]})

    leaf_matcher_recur(clade_root, all_syn_seqs, all_nat_seqs_dict)

    return pd.DataFrame(output)

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_hmmer", action="store", dest="input_hmmer",
                    help="unprocessed hmmer table")

parser.add_argument("-T", "--tree", action="store", dest="tree",
                    help="input protein family tree")

parser.add_argument("--J_params", action="store", dest="J_params",
                    help="bmDCA J params")

parser.add_argument("--h_params", action="store", dest="h_params",
                    help="bmDCA h parameters")

parser.add_argument("-M", "--simulated_MSA", action="store", dest="simulated_MSA",
                    help="MSA resulting from simulation")

parser.add_argument("--original_MSA_seed", action="store", dest="original_MSA_seed",
                    help="original seed MSA used for simulation")

parser.add_argument("--original_MSA_full", action="store", dest="original_MSA_full",
                    help="original seed MSA used for simulation")

parser.add_argument("-O", "--output", action="store", dest="output", 
                    help="processed hmmer table")

parser.add_argument("--no_phylogeny", action="store_true", dest="no_phylogeny",
                    help="do not evolve along a tree")

args = parser.parse_args()

input_hmmer = args.input_hmmer
output = args.output
simulated_MSA = args.simulated_MSA
original_MSA_seed = args.original_MSA_seed
original_MSA_full = args.original_MSA_full
J_params = args.J_params
h_params = args.h_params
tree_path = args.tree
no_phylogeny = args.no_phylogeny

if not no_phylogeny:
    tree = Phylo.read(tree_path,"newick")
    tree.root_at_midpoint()

table = open(input_hmmer)
with open(output,"w") as f:
    line = table.readline()
    while line: 
        if not line.startswith("#"):
            
            f.writelines(line)
            
        line = table.readline()

relevant_cols = ["sequence_name","hmmer_seq_score"]
scores_table = pd.read_csv(output, delimiter="\s+",header=None, usecols=[0,5], names=relevant_cols)

synth_sequences = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(simulated_MSA, "fasta")]
nat_seed_sequences = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(original_MSA_seed, "fasta")]
nat_full_sequences = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(original_MSA_full, "fasta")]


n_cols = len(synth_sequences[0][1])
n_rows = len(synth_sequences)

if not no_phylogeny:
    nat_seed_sequences_dict = dict(nat_seed_sequences)
    matched_seqs = leaf_matcher(tree.clade, all_syn_seqs=synth_sequences, all_nat_seqs_dict=nat_seed_sequences_dict)
    synth_sequences = pd.DataFrame(synth_sequences, columns=["sequence_name","sequence"])
    scores_table = scores_table.merge(synth_sequences, on="sequence_name").merge(matched_seqs, on="sequence_name")
    scores_table = scores_table[["sequence_name","sequence","corr_nat_seq_name","corr_nat_seq","hmmer_seq_score"]]
else:
    synth_sequences = pd.DataFrame(synth_sequences, columns=["sequence_name","sequence"])
    scores_table = scores_table.merge(synth_sequences, on="sequence_name")
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
nat_seed_sequences_array = np.array([list(seq) for _,seq in nat_seed_sequences], dtype=np.bytes_).view(np.uint8)
nat_full_sequences_array = np.array([list(seq) for _,seq in nat_full_sequences], dtype=np.bytes_).view(np.uint8)

distance_matrix_seed = cdist(sim_sequences_array, nat_seed_sequences_array, "hamming")
distance_matrix_full = cdist(sim_sequences_array, nat_full_sequences_array, "hamming")

min_natural_ham_distance_seed = list(distance_matrix_seed.min(axis = 1))
max_natural_ham_distance_seed = list(distance_matrix_seed.max(axis = 1))

min_natural_ham_distance_full = list(distance_matrix_full.min(axis = 1))
max_natural_ham_distance_full = list(distance_matrix_full.max(axis = 1))

scores_table["min_natural_ham_dist_seed"] = min_natural_ham_distance_seed
scores_table["max_natural_ham_dist_seed"] = max_natural_ham_distance_seed

scores_table["min_natural_ham_dist_full"] = min_natural_ham_distance_full
scores_table["max_natural_ham_dist_full"] = max_natural_ham_distance_full

self_distance_matrix = cdist(sim_sequences_array, sim_sequences_array, "hamming")
max_self_ham_distance = list(self_distance_matrix.max(axis = 1))

scores_table["max_self_ham_distance"] = max_self_ham_distance

scores_table.to_csv(output, sep="\t", index = False)



    




