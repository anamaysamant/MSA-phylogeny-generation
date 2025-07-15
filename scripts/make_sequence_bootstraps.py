import numpy as np
from aux_msa_functions import *
from Bio import Phylo

input_MSA = snakemake.input[0]
seed = int(snakemake.wildcards["sim_ind"])


np.random.seed(seed)

nat_seed_sequences = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(input_MSA, "fasta")]

seqs = [sequence[1] for sequence in nat_seed_sequences]
n_seqs = len(seqs)

row_sample_inds = np.random.choice(range(n_seqs), n_seqs, replace=True)

bootstrap_MSA = [(f"seq{j}", nat_seed_sequences[i][1]) for j,i in enumerate(row_sample_inds)]

Seq_tuples_to_fasta(bootstrap_MSA, snakemake.output[0])