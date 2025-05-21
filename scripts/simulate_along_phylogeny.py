from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string
from pathlib import Path
import sys
import esm

import numpy as np
from Bio import SeqIO, Phylo
import pandas as pd
from scipy.spatial.distance import squareform, pdist, cdist
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
from time import time
import os
import sys

import argparse

from aux_msa_functions import *
from MSAGeneratorESM import MSAGeneratorESM
from MSAGeneratorPottsModel import MSAGeneratorPottsModel

try:
    from MSAGeneratorESMC import MSAGeneratorESMC
except:
    pass

from select_gpu import get_free_gpu


import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu()) 

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--tool", action="store", dest="tool",
                    help="tool used for simulation of MSA")

parser.add_argument("-M", "--input_MSA", action="store", dest="input_MSA",
                    help="input protein family seed MSA")

parser.add_argument("--input_MSA_full", action="store", dest="input_MSA_full",
                    help="input protein family full MSA")

parser.add_argument("-T", "--input_tree", action="store", dest="input_tree",
                    help="input protein family tree")

parser.add_argument("-O", "--output", action="store", dest="output",
                    help="location of simulated MSA"
                )

parser.add_argument("-n", "--num_seqs", action="store", dest="num_seqs",
                    help="number of sequences to simulate", type=int
                )

parser.add_argument("--J_params", action="store", dest="J_params",
                    help="bmDCA J params")

parser.add_argument("--h_params", action="store", dest="h_params",
                    help="bmDCA h parameters")

parser.add_argument("-s", "--start_seq_index", action="store", dest="start_seq_index",
                    help="index in MSA of starting sequence of simulation", type=int, default=0
                )

parser.add_argument("-c", "--context_type", action="store", dest="context_type",
                    help="define the type of context to use when using MSA transformer for simulation")

parser.add_argument("-cs", "--context_size", action="store", dest="context_size",
                    help="size of context for MSA-1b simulation along phylogeny", type=int)

parser.add_argument("--context_sampling", action="store", dest="context_sampling",
                    help="method of sampling context for MSA-1b")

parser.add_argument("--proposal_type", action="store", dest="proposal_type",
                    help="proposal distribution used")

parser.add_argument("--chunked", action="store_true", dest="chunked",
                    help="generate chunks of new MSA")

parser.add_argument("--no_phylogeny", action="store_true", dest="no_phylogeny",
                    help="do not evolve along a tree")

parser.add_argument( "--n_mutations", action="store", dest="n_mutations", 
                    help="number of mutations for independent MCMC sequence generation", type=int)

parser.add_argument( "--FT_fam", action="store", dest="FT_fam", 
                    help="family on which MSA transformer is finetuned")

parser.add_argument( "--n_sequences", action="store", dest="n_sequences", 
                    help="number of independent sequences to generate via MCMC", type=int)

parser.add_argument( "--log_file", action="store", dest="log_file", 
                    help="log file for the simulation")

parser.add_argument( "--seed", action="store", dest="seed", 
                    help="random seed to use", type=int, default=0)

parser.add_argument( "--r_eff", action="store", dest="r_eff", 
                    help="random seed to use", type=float,default=1.0)


args = parser.parse_args()

tool = args.tool
MSA_path_seed = args.input_MSA
MSA_path_full = args.input_MSA_full
tree_path = args.input_tree
num_seqs = args.num_seqs
context_type = args.context_type
context_size = args.context_size
context_sampling = args.context_sampling
proposal_type = args.proposal_type
chunked = args.chunked
starting_seq_index = args.start_seq_index
output = args.output
J_params = args.J_params
h_params = args.h_params
no_phylogeny = args.no_phylogeny
n_mutations = args.n_mutations
n_sequences = args.n_sequences
seed = args.seed
FT_fam = args.FT_fam
log_file = args.log_file
r_eff = args.r_eff

work_dir = os.getcwd()
os.chdir("./scripts")
from MSA_phylogeny_class import Creation_MSA_Generation_MSA1b_Cython

if tool == "ProtMamba":
    from ProtMamba_Phylogeny_class import ProtMamba_Simulator
    
os.chdir(work_dir)

logging.basicConfig(
    filename=log_file,               
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode="a"
)

np.random.seed(seed)


all_seqs_seed = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(MSA_path_seed, "fasta")]

if MSA_path_full != None:
    all_seqs_full = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(MSA_path_full, "fasta")]

if num_seqs == None:
    num_seqs = len(all_seqs_seed)

if not no_phylogeny:
    tree = Phylo.read(tree_path,"newick")
    tree.root_at_midpoint()

if tool == "MSA_1b":

    t1 = time()

    _, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()

    if FT_fam != None:
        model_to_use = torch.load(f"./finetuned_MSA_models/MSA_finetuned_{FT_fam}.pt")

    else:
        model_to_use, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()

    if not no_phylogeny:
        MSA_gen_obj = Creation_MSA_Generation_MSA1b_Cython(MSA = all_seqs_seed, full_tree = tree, full_tree_path = tree_path, 
                                                         start_seq_index=starting_seq_index, model_to_use=model_to_use, alphabet=alphabet)
    else:
        MSA_gen_obj = Creation_MSA_Generation_MSA1b_Cython(MSA = all_seqs_seed, start_seq_index=starting_seq_index, model_to_use=model_to_use, alphabet=alphabet)

    method = "minimal"
    masked = True
    chunked = args.chunked

    if not no_phylogeny:

        if context_type == "dynamic":
            logging.info(f"Simulating along phylogeny using MSA-1b ({context_type}-{context_sampling}-{context_size}-{proposal_type})")
        else:
            logging.info(f"Simulating along phylogeny using MSA-1b ({context_type}-{context_size}-{proposal_type})")

        if not chunked:
            new_MSA = MSA_gen_obj.msa_tree_phylo(tree.clade,flip_before_start=0, method=method, 
            masked=masked, context_type = context_type, context_sampling = context_sampling, context_size = context_size, proposal = proposal_type, r_eff = r_eff)
        else:
            new_MSA = MSA_gen_obj.msa_tree_phylo_chunked(total_sequences = 100,sequences_per_iteration = 10, method=method, masked=masked, proposal = proposal_type)

        t2 = time()



        if context_type == "dynamic":
            logging.info(f"Time taken to simulate along phylogeny using MSA-1b ({context_type}-{context_sampling}-{context_size}-{proposal_type}): {(t2-t1)/60} minutes")
        else:
            logging.info(f"Time taken to simulate along phylogeny using MSA-1b ({context_type}-{context_size}-{proposal_type}): {(t2-t1)/60} minutes")
    else:
        new_MSA = MSA_gen_obj.msa_no_phylo(context_size = context_size, n_sequences = n_sequences,n_mutations = n_mutations, method=method, 
                                           masked=masked, proposal = proposal_type)

    Seq_tuples_to_fasta(new_MSA,output)

    diagnostics_folder = 'diagnostics/' + '/'.join(output.split("/")[1:-1]) + f"/{seed}"

    if not os.path.exists(diagnostics_folder):
        os.makedirs(diagnostics_folder)

    if proposal_type == "random":
        np.save(os.path.join(diagnostics_folder,"hamming_distances.npy"),MSA_gen_obj.hamming_distances)
        np.save(os.path.join(diagnostics_folder,"mean_proposal_probs.npy"),MSA_gen_obj.mean_proposal_probs)
        np.save(os.path.join(diagnostics_folder,"n_mutations.npy"),MSA_gen_obj.n_mutations)
        np.save(os.path.join(diagnostics_folder,"n_proposals.npy"),MSA_gen_obj.n_proposals)
        np.save(os.path.join(diagnostics_folder,"mean_accepted_proposal_probs.npy"),MSA_gen_obj.mean_accepted_proposal_probs)
        np.save(os.path.join(diagnostics_folder,"mean_acceptance_criteria.npy"),MSA_gen_obj.mean_acceptance_criteria)
        np.save(os.path.join(diagnostics_folder,"mean_accepted_acceptance_criteria.npy"),MSA_gen_obj.mean_accepted_acceptance_criteria)
    elif proposal_type == "msa_prob_dist":
        np.save(os.path.join(diagnostics_folder,"mean_proposal_probs.npy"),MSA_gen_obj.mean_proposal_probs)
        np.save(os.path.join(diagnostics_folder,"n_mutations.npy"),MSA_gen_obj.n_mutations)
        np.save(os.path.join(diagnostics_folder,"hamming_distances.npy"),MSA_gen_obj.hamming_distances)
        
elif tool == "ESM2":

    t1 = time()

    ESM_gen_obj = MSAGeneratorESM(number_of_nodes= len(all_seqs_seed[0][1]), number_state_spin=21,
                                batch_size=1,model="facebook/esm2_t33_650M_UR50D")
    
    if starting_seq_index == -1:
        char_list = list("-ACDEFGHIKLMNPQRSTVWY")
        rand_char_order = np.random.choice(range(len(char_list)), len(all_seqs_seed[0][1]), replace = True)
        first_sequence = [char_list[i] for i in rand_char_order]
        first_sequence = ''.join(first_sequence)
    else:
        first_sequence = all_seqs_seed[starting_seq_index][1]
    
    new_MSA = ESM_gen_obj.msa_tree_phylo(clade_root=tree.clade, first_sequence=first_sequence, flip_before_start=0, proposal = proposal_type)

    seq_records = []
    for i in range(new_MSA.shape[0]):
        seq_records.append(SeqRecord(seq=Seq(''.join(ESM_gen_obj.inverse_amino_acid_map[index]
                                                    for index in new_MSA[i])),
                                    id='seq' + str(i), description='seq' + str(i), name='seq' + str(i)))
    SeqIO.write(seq_records,output, "fasta")

    t2 = time()

    logging.info(f"Time taken to simulate along phylogeny using ESM2: {(t2-t1)/60} minutes")

elif tool == "ESMC":

    t1 = time()

    ESMC_gen_obj = MSAGeneratorESMC(number_of_nodes= len(all_seqs_seed[0][1]), number_state_spin=21,model="esmc_300m")
    
    if starting_seq_index == -1:
        char_list = list("-ACDEFGHIKLMNPQRSTVWY")
        rand_char_order = np.random.choice(range(len(char_list)), len(all_seqs_seed[0][1]), replace = True)
        first_sequence = [char_list[i] for i in rand_char_order]
        first_sequence = ''.join(first_sequence)
    else:
        first_sequence = all_seqs_seed[starting_seq_index][1]
    
    new_MSA = ESMC_gen_obj.msa_tree_phylo(clade_root=tree.clade, first_sequence=first_sequence, flip_before_start=0, proposal = proposal_type)

    seq_records = []
    for i in range(new_MSA.shape[0]):
        seq_records.append(SeqRecord(seq=Seq(''.join(ESMC_gen_obj.inverse_amino_acid_map[index]
                                                    for index in new_MSA[i])),
                                    id='seq' + str(i), description='seq' + str(i), name='seq' + str(i)))
    SeqIO.write(seq_records,output, "fasta")

    t2 = time()

    logging.info(f"Time taken to simulate along phylogeny using ESMC: {(t2-t1)/60} minutes")

elif tool == "Potts":

    t1 = time()

    J_params = np.load(J_params)
    h_params = np.load(h_params)

    Potts_gen_obj = MSAGeneratorPottsModel(field = h_params, coupling = J_params)
    
    if starting_seq_index == -1:
        print("x")
        char_list = list("-ACDEFGHIKLMNPQRSTVWY")
        rand_char_order = np.random.choice(range(len(char_list)), len(all_seqs_seed[0][1]), replace = True)
        first_sequence = [char_list[i] for i in rand_char_order]
        first_sequence = ''.join(first_sequence)
    else:
        first_sequence = all_seqs_seed[starting_seq_index][1]

    first_sequence = np.array([Potts_gen_obj.bmdca_mapping[char] for char in list(first_sequence)])

    if no_phylogeny:
        new_MSA = Potts_gen_obj.msa_no_phylo(n_sequences=n_sequences, n_mutations=n_mutations, first_sequence=first_sequence)
    else:
        new_MSA = Potts_gen_obj.msa_tree_phylo(clade_root=tree.clade, first_sequence=first_sequence, proposal = None, flip_before_start=0)
    
    seq_records = []
    for i in range(new_MSA.shape[0]):
        seq_records.append(SeqRecord(seq=Seq(''.join(Potts_gen_obj.bmdca_mapping_inv[index]
                                                    for index in new_MSA[i])),
                                    id='seq' + str(i), description='seq' + str(i), name='seq' + str(i)))
    SeqIO.write(seq_records,output, "fasta")

    t2 = time() 

    logging.info(f"Time taken to simulate along phylogeny using Potts: {(t2-t1)/60} minutes")

elif tool == "ProtMamba":

    t1 = time()

    if FT_fam != None:
        model_to_use = torch.load(f"./finetuned_ProtMamba_models/MSA_finetuned_{FT_fam}.pt")

    else:
        model_to_use = None

    if not no_phylogeny:
        ProtMamba_obj = ProtMamba_Simulator(MSA = all_seqs_seed, full_tree = tree, full_tree_path = tree_path, 
                                                         start_seq_index=starting_seq_index, model_to_use=model_to_use)
    else:
        ProtMamba_obj = ProtMamba_Simulator(MSA = all_seqs_seed, start_seq_index=starting_seq_index, model_to_use=model_to_use)

    if not no_phylogeny:

        if context_type == "dynamic":
            logging.info(f"Simulating along phylogeny using ProtMamba ({context_type}-{context_sampling}-{context_size}-{proposal_type})")
        else:
            logging.info(f"Simulating along phylogeny using ProtMamba ({context_type}-{context_size}-{proposal_type})")

        new_MSA = ProtMamba_obj.mamba_tree_phylo(tree.clade,flip_before_start=0, context_type = context_type, context_sampling = context_sampling, context_size = context_size, proposal = proposal_type)

        t2 = time()

        if context_type == "dynamic":
            logging.info(f"Time taken to simulate along phylogeny using ProtMamba ({context_type}-{context_sampling}-{context_size}-{proposal_type}): {(t2-t1)/60} minutes")
        else:
            logging.info(f"Time taken to simulate along phylogeny using ProtMamba ({context_type}-{context_size}-{proposal_type}): {(t2-t1)/60} minutes")
    else:
        new_MSA = ProtMamba_obj.mamba_no_phylo(context_size = context_size, n_sequences = n_sequences,n_mutations = n_mutations, proposal = proposal_type)

    Seq_tuples_to_fasta(new_MSA,output)