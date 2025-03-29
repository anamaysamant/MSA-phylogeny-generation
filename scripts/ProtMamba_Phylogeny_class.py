from ProtMamba_ssm.utils import *
from ProtMamba_ssm.dataloaders import *
from ProtMamba_ssm.modules import *
from aux_msa_functions import *

import numpy as np
import torch
import os
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
from Bio import SeqIO, Phylo
import sys

class ProtMamba_Simulator:
    
    def __init__(self, MSA, model_to_use = None, start_seq_index = 0,full_tree = None, full_tree_path = None):

        torch.cuda.empty_cache()

        self.original_MSA = MSA
        self.start_seq_name = MSA[0][0]
        self.full_tree = full_tree
        self.full_tree_path = full_tree_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.n_rows = len(self.original_MSA)
        self.n_cols = len(self.original_MSA[0][1])

        self.context = None
        self.context_size = None

        if model_to_use == None:

            self.model = load_model("./ProtMamba-Long-foundation",
                        model_class=MambaLMHeadModelwithPosids,
                        device="cuda",
                        dtype=torch.bfloat16,
                        checkpoint_mixer=False # Must be False when using model for Inference
                        )
            
            self.model = self.model.to(self.device)
    

        if start_seq_index == -1:
            char_list = list("-ACDEFGHIKLMNPQRSTVWY")
            rand_char_order = np.random.choice(range(len(char_list)), self.n_cols, replace = True)
            rand_seq = [char_list[i] for i in rand_char_order]
            rand_seq = ''.join(rand_seq)
            self.original_MSA = [("rand_seq",rand_seq)] + self.original_MSA

        self.init_seq = tokenizer([self.original_MSA[0][1]], concatenate=True)
        self.init_seq = self.init_seq.to(self.device)

        self.start_seq_init(start_seq_index)
    

    def sample_static_context(self, all_sequences, method, context_size):
            
        if context_size > 0:

            if method == "greedy":

                context = greedy_select(all_sequences, num_seqs = context_size + 1, random_start = False)
                context = context[1:]
                self.context_size = len(context)

                self.context = tokenizer([seq[1] for seq in context], concatenate=True)
                self.context = self.context.to(self.device)
            
            elif method == "random":

                random_ind = list(np.random.choice(range(1,len(all_sequences)),context_size, replace = False))
                context = [all_sequences[i] for i in random_ind]
                self.context_size = len(context)

                self.context = tokenizer([seq[1] for seq in all_sequences], concatenate=True)
                self.context = self.context.to(self.device)

    def start_seq_init(self, start_seq_index = 0):

        if start_seq_index != 0 and start_seq_index != -1:

            start_seq = self.original_MSA[start_seq_index]
            del self.original_MSA[start_seq_index]
            self.original_MSA = [start_seq] + self.original_MSA

            self.start_seq_name = self.original_MSA[0][0]
            self.init_seq = tokenizer([self.original_MSA[0][1]], concatenate=True)
            self.init_seq = self.init_seq.to(self.device)
    

    def mamba_tree_phylo(self, clade_root, context_type, 
                        context_size, context_sampling, proposal, flip_before_start = 0):
                
        self.phylogeny_MSA = []
        
        self.sample_static_context(self.original_MSA, method = "random", context_size = context_size)
        first_sequence_tokens = self.mcmc(Number_of_Mutation = flip_before_start, previous_sequence_tokens = self.init_seq.clone(),
                                        proposal = proposal)
    
        if context_type == "static":
            self.mamba_tree_phylo_recur(clade_root, first_sequence_tokens.clone(), proposal = proposal)

        elif context_type == "dynamic":
            self.mamba_tree_phylo_recur_dynamic(clade_root, first_sequence_tokens.clone(),context_sampling=context_sampling, context_size = context_size, proposal = proposal)

        results = self.phylogeny_MSA.copy()
        self.phylogeny_MSA = []
        self.context = None
        self.init_seq = None
        
        return results
    
    def mamba_tree_phylo_recur(self, clade_root, previous_sequence_tokens, proposal):
        
        b = clade_root.clades

        if len(b)>0:
            for clade in b:

                n_mutations = int(clade.branch_length*self.n_cols)
                new_sequence_tokens = self.mcmc(n_mutations, previous_sequence_tokens.clone(), proposal)
                self.mamba_tree_phylo_recur(clade, new_sequence_tokens.clone(), proposal=proposal)
        else:

            final_seq = ""
            for i in range(1,previous_sequence_tokens.shape[1]):

                char_index = int(previous_sequence_tokens[0,i].cpu().numpy())
                character = ID_TO_AA[char_index]
                final_seq += character
                            
            seq_index = len(self.phylogeny_MSA)
            self.phylogeny_MSA.append((f"seq{seq_index}",final_seq))

            print(f"Number of sequences generated: {len(self.phylogeny_MSA)}")
            
    def mamba_tree_phylo_recur_dynamic(self, clade_root, previous_sequence_tokens, context_size, context_sampling, proposal):
        
        b = clade_root.clades
        
        if len(b)>1:
            for clade in b:
                # print("entering new branch")
                n_mutations = int(clade.branch_length*self.n_cols)
                desc_leaves = [node.name for node in clade.get_terminals() if node.name != self.start_seq_name]
                desc_sequences = [elem for elem in self.original_MSA if elem[0] in desc_leaves]
                if len(desc_leaves) > context_size:

                    if context_sampling == "random":
                        
                        random_ind = list(np.random.choice(range(len(desc_sequences)),context_size, replace = False))
                        self.context = [desc_sequences[i] for i in random_ind]

                    elif context_sampling == "greedy":
                        
                        self.context = greedy_select(desc_sequences, num_seqs = context_size)
              
                    self.context_size = len(self.context)
                    self.context = tokenizer([seq[1] for seq in self.context], concatenate=True)
                    self.context = self.context.to(self.device)
                
                # new_tree = self.generate_subtree(desc_leaves)
                new_sequence_tokens = self.mcmc(n_mutations, previous_sequence_tokens.clone(), proposal)
                self.mamba_tree_phylo_recur_dynamic(clade, new_sequence_tokens.clone(),context_size = context_size, 
                                                  context_sampling = context_sampling, proposal = proposal)
        else:

            final_seq = ""
            for i in range(1,previous_sequence_tokens.shape[1]):

                char_index = int(previous_sequence_tokens[0,i].cpu().numpy())
                character = ID_TO_AA[char_index]
                final_seq += character
                            
            seq_index = len(self.phylogeny_MSA)
            self.phylogeny_MSA.append((f"seq{seq_index}",final_seq))
   
            print(f"Number of sequences generated: {len(self.phylogeny_MSA)}")


    def mamba_no_phylo(self, context_size, n_sequences, n_mutations, proposal):

        syn_sequences_list = []

        self.sample_static_context(self.original_MSA, method = "greedy", context_size = context_size)

        for i in range(n_sequences):

            new_sequence_tokens = self.mcmc(n_mutations, self.init_seq.clone(), proposal)

            final_seq = ""
            for i in range(1, new_sequence_tokens.shape[1]):

                char_index = int(new_sequence_tokens[0,0,i].cpu().numpy())
                character = self.model_alphabet_mapping_inv[char_index]
                final_seq += character
                            
            seq_index = len(syn_sequences_list)
            syn_sequences_list.append((f"seq{seq_index}",final_seq))

        return syn_sequences_list
    
    def mcmc(self, Number_of_Mutation, previous_sequence_tokens, proposal):  
    
        c_mutation = 0

        print(f"Number of mutations: {Number_of_Mutation}")
        if proposal == "logits":

            while c_mutation<Number_of_Mutation:
      
                selected_pos =  np.random.randint(1, self.n_cols + 1)

                if selected_pos > previous_sequence_tokens.shape[1] - 1:
                    continue
                
                mask_dictionary = {"<mask-1>": ((selected_pos,selected_pos + 1),1)}
      
                input_seq, targ_pos, is_fim_dict = prepare_target(previous_sequence_tokens, use_fim=mask_dictionary)

                input_seq = input_seq.to(self.device)
                targ_pos = targ_pos.to(self.device)


                context_tokens, context_pos_ids = prepare_tokens(self.context,
                                    target_tokens=input_seq,
                                    target_pos_ids=targ_pos,
                                    DatasetClass=Uniclust30_Dataset,
                                    num_sequences=self.context_size,
                                    fim_strategy="no-scramble",
                                    mask_fraction=1,
                                    max_patches=1,
                                    add_position_ids="1d")  
                
                # context_tokens, context_pos_ids, tokens_fim, pos_ids_fim, is_fim_dict = prepare_dataset_for_fim_generation(context_tokens, context_pos_ids)
  
                output = generate_sequence(self.model,
                            context_tokens,
                            position_ids=context_pos_ids,
                            is_fim=is_fim_dict,
                            max_length=context_tokens.shape[1] + len(self.original_MSA[0][1]),
                            temperature=1.,
                            top_k=10,
                            top_p=0.0,
                            return_dict_in_generate=True,
                            output_scores=True,
                            eos_token_id=AA_TO_ID["<cls>"],
                            device="cuda")

                output_seq = output["generated"][0]

                translation_table = dict.fromkeys(map(ord, '<>-clsmask12345'), None)
                try:
                    output_char = output_seq.translate(translation_table)[0]
                except:
                    continue

                if output_char not in list("-ACDEFGHIKLMNPQRSTVWY"):
                    continue

                if AA_TO_ID[output_char] == previous_sequence_tokens[0,selected_pos]:
                    continue

                assert input_seq[0,selected_pos] == 33,'<mask> token should be present'

                previous_sequence_tokens[0,selected_pos] = AA_TO_ID[output_char]

                assert (previous_sequence_tokens == 33).sum() == 0,'<mask> token should be present'

                # logits = output["scores"] 

                c_mutation += 1
                
                # previous_sequence_tokens = tokenizer([output_seq], concatenate=True)
                # previous_sequence_tokens = previous_sequence_tokens.to(self.device)
        
        return previous_sequence_tokens



    