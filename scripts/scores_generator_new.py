import argparse
import pandas as pd
from aux_msa_functions import *
import time
from scipy.spatial.distance import cdist
from Bio import Phylo, Align
import os
import subprocess

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

AA_3_letters = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
AA_1_letter = list("ARNDCQEGHILKMFPSTWYV")

AA_mapping = {k:v for k,v in zip(AA_3_letters,AA_1_letter)}

families = ["PF00271","PF00005","PF00004","PF01535","PF00595","PF00397","PF00153","PF07679",
            "PF00076","PF00072","PF00096","PF00512","PF00041","PF13354","PF02518",
            "PF01356","PF03440","PF04008","PF06351","PF06355", "PF16747","PF18648"]

families = ["PF00005"]

sim_inds = list(range(1,11))

for family in families:

    print(family)
        
    for sim_ind in sim_inds:

        # simulated_MSA = f"../data/msa-seed-simulations/MSA-1b/{family}/init-seq-0/logits-proposal/static-context/10/{family}-{sim_ind}.fasta"
        simulated_MSA = f"../data/sequences-bootstrap-seed-msa/{family}/bootstrap-{sim_ind}.fasta"
        original_MSA_seed = f"../data/protein-families-msa-seed/{family}_seed.fasta"
        original_MSA_full = f"../data/protein-families-msa-full/{family}.fasta"
        # output = f"../scores/msa-seed-simulations/MSA-1b/{family}/init-seq-0/logits-proposal/static-context/10/{family}-{sim_ind}-dist.tsv"
        output_dir = f"../scores/sequences-bootstrap-seed-msa-dist/{family}/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output = f"../scores/sequences-bootstrap-seed-msa-dist/{family}/bootstrap-{sim_ind}.tsv"
        tree_path = f"../data/seed-trees/{family}_seed.newick"
        
        tree = Phylo.read(tree_path,"newick")
        tree.root_at_midpoint()

        synth_sequences = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(simulated_MSA, "fasta")]
        nat_seed_sequences = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(original_MSA_seed, "fasta")]
        nat_full_sequences = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(original_MSA_full, "fasta")]


        n_cols = len(synth_sequences[0][1])
        n_rows = len(synth_sequences)

        scores_table = pd.DataFrame()

        nat_seed_sequences_dict = dict(nat_seed_sequences)        
        matched_seqs = leaf_matcher(tree.clade, all_syn_seqs=synth_sequences, all_nat_seqs_dict=nat_seed_sequences_dict)
        synth_sequences = pd.DataFrame(synth_sequences, columns=["sequence_name","sequence"])

        scores_table = synth_sequences.merge(matched_seqs, on="sequence_name")

        sim_sequences_array = np.array([list(seq) for seq in scores_table["sequence"]], dtype=np.bytes_).view(np.uint8)
        nat_seed_sequences_array = np.array([list(seq) for _,seq in nat_seed_sequences], dtype=np.bytes_).view(np.uint8)
        nat_full_sequences_array = np.array([list(seq) for _,seq in nat_full_sequences], dtype=np.bytes_).view(np.uint8)

        distance_matrix_seed = cdist(sim_sequences_array, nat_seed_sequences_array, "hamming")
        
        
        partitioned_mat = np.partition(distance_matrix_seed, kth=1, axis = 1)

        print(partitioned_mat)

        second_min_natural_ham_distance_seed = list(partitioned_mat[:,1])
        scores_table["second_min_natural_ham_distance_seed"] = second_min_natural_ham_distance_seed

        try:
            distance_matrix_full = cdist(sim_sequences_array, nat_full_sequences_array, "hamming")
            partitioned_mat = np.partition(distance_matrix_full, kth=1, axis = 1)
            second_min_natural_ham_distance_full = list(np.partition(distance_matrix_full, kth=1, axis = 1)[:,1])
            scores_table["second_min_natural_ham_distance_full"] = second_min_natural_ham_distance_full
        except:
            pass

        self_distance_matrix = cdist(sim_sequences_array, sim_sequences_array, "hamming")
        max_self_ham_distance = list(self_distance_matrix.max(axis = 1))
        min_self_ham_distance = list(np.partition(self_distance_matrix, kth=1, axis = 1)[:,1])
        mean_self_ham_distance = list(self_distance_matrix.mean(axis = 1))


        scores_table["min_self_ham_distance"] = min_self_ham_distance
        scores_table["mean_self_ham_distance"] = mean_self_ham_distance

        scores_table.to_csv(output, sep="\t", index = False)



        




