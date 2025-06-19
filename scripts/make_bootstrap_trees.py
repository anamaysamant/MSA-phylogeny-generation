import numpy as np
from aux_msa_functions import *
from Bio import Phylo

def reorder_seqs(tree_root, all_nat_seqs_dict):

    output = []
    
    def reorder_seqs_recur(tree_root, all_nat_seqs_dict):
    
        b = tree_root.clades
        
        if len(b)>0:
            for clade in b:
               reorder_seqs_recur(clade, all_nat_seqs_dict) 
        else:
            counter = len(output)
            output.append((tree_root.name,all_nat_seqs_dict[tree_root.name]))

    reorder_seqs_recur(tree_root, all_nat_seqs_dict)

    return output

input_MSA = snakemake.input[0]
seed_tree_path = snakemake.input[1]
seed = int(snakemake.wildcards["sim_ind"])


np.random.seed(seed)

nat_seed_sequences = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(input_MSA, "fasta")]

tree_biophylo = Phylo.read(seed_tree_path, "newick")
tree_biophylo.root_at_midpoint()

ordered_seqs = reorder_seqs(tree_biophylo.clade, dict(nat_seed_sequences))

array_MSA = np.array([list(sequence[1]) for sequence in ordered_seqs])
len_MSA = array_MSA.shape[1]
n_seqs = array_MSA.shape[0] 

column_sample_inds = np.random.choice(range(len_MSA), len_MSA, replace=True)

bootstrap_MSA = array_MSA[:,column_sample_inds]
bootstrap_MSA = [(nat_seed_sequences[i][0], ''.join(bootstrap_MSA[i])) for i in range(n_seqs)]

Seq_tuples_to_fasta(bootstrap_MSA, snakemake.output[0])