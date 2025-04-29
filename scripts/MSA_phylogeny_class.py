import numpy as np
import torch
import esm
from Bio import Phylo
import os
from scipy.spatial.distance import squareform, pdist, cdist
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable

from aux_msa_functions import greedy_select

class Creation_MSA_Generation_MSA1b_Cython:
    
    def __init__(self, MSA, model_to_use = None, alphabet = None, start_seq_index = 0,full_tree = None, full_tree_path = None, random_init_seq = False, seed = None):

        torch.cuda.empty_cache()

        self.seed = seed
        self.original_MSA = MSA
        self.start_seq_name = MSA[0][0]
        self.full_tree = full_tree
        self.full_tree_path = full_tree_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.n_rows = len(self.original_MSA)
        self.n_cols = len(self.original_MSA[0][1])
        
        if model_to_use == None:
            self.model, self.alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        else:
            self.model = model_to_use
            self.alphabet = alphabet
            del model_to_use, alphabet

        self.model = self.model.to(self.device)


        self.model_alphabet_mapping = self.alphabet.to_dict()
        self.model_alphabet_mapping_inv = dict(zip(range(len(self.alphabet.all_toks)), self.alphabet.all_toks))
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval() 

        self.full_context = self.original_MSA[1:]

        self.context = None

        if start_seq_index == -1:
            print("starting from a random sequence")
            char_list = list("-ACDEFGHIKLMNPQRSTVWY")
            rand_char_order = np.random.choice(range(len(char_list)), self.n_cols, replace = True)
            rand_seq = [char_list[i] for i in rand_char_order]
            rand_seq = ''.join(rand_seq)
            self.original_MSA = [("rand_seq",rand_seq)] + self.original_MSA
        
        _,_,self.init_seq = self.batch_converter([self.original_MSA[:1]])
        self.init_seq = self.init_seq.to(self.device)

        self.start_seq_init(start_seq_index)

        self.hamming_distances = []
        self.mean_proposal_probs = []
        self.mean_accepted_proposal_probs = []
        self.n_mutations = []
        self.n_proposals = []
        self.mean_acceptance_criteria = []
        self.mean_accepted_acceptance_criteria = []
    
    def sample_static_context(self, all_sequences, method, context_size):
            
        if context_size > 0:

            if method == "greedy":

                context = greedy_select(all_sequences, num_seqs = context_size + 1, seed = self.seed, random_start=False)
                context = context[1:]

                _,_,self.context = self.batch_converter([context])
                self.context = self.context.to(self.device)
            
            elif method == "random":

                random_ind = list(np.random.choice(range(1,len(all_sequences)),context_size, replace = False))
                context = [all_sequences[i] for i in random_ind]

                _,_,self.context = self.batch_converter([context])
                self.context = self.context.to(self.device)

    def start_seq_init(self, start_seq_index = 0):

        if start_seq_index != 0 and start_seq_index != -1:

            start_seq = self.original_MSA[start_seq_index]
            del self.original_MSA[start_seq_index]
            self.original_MSA = [start_seq] + self.original_MSA

            self.start_seq_name = self.original_MSA[0][0]
            _,_,self.init_seq = self.batch_converter([self.original_MSA[:1]])
            self.init_seq = self.init_seq.to(self.device)
    
    def prob_calculator(self, batch_tokens, selected_pos, method, masked):
        
        softmax = torch.nn.Softmax(dim = -1)

        batch_tokens_copy = batch_tokens.clone()
        original_char_index = batch_tokens[0,0,selected_pos] 
        
        if masked == True:
            batch_tokens_copy[0,0,selected_pos] = self.model_alphabet_mapping["<mask>"]
        
        with torch.no_grad(): 

            original_char_prob = (softmax(self.model(batch_tokens_copy)["logits"][0,0,selected_pos,:]).cpu().numpy())[original_char_index]
        
            if method == "minimal":

                return np.log(original_char_prob)
                  
            if method == "full" or method == "row":

                # probs = softmax(self.msa_lm_head(self.msa_light(batch_tokens_copy)[0,0,:,:])).cpu().numpy()
                probs = softmax(self.model(batch_tokens_copy)["logits"][0,0,:,:]).cpu().numpy()
                          
                log_prob_row = 0
                
                for i in range(1,self.n_cols +1):
                    char_index = batch_tokens[0,0,i]
                    log_prob_row += np.log(probs[i,char_index])
    
                if method == "row":
    
                    return log_prob_row
                                               
            if method == "full" or method == "col":

                log_prob_col = 0
    
                probs = softmax(self.model(batch_tokens_copy)["logits"][0,:,selected_pos,:]).cpu().numpy()

                for i in range(self.n_rows):
                    char_index = batch_tokens[0,i,selected_pos]
                    log_prob_col += np.log(probs[i,char_index])
    
                if method == "col":
    
                    return log_prob_col
    
            log_total_prob = log_prob_row + log_prob_col - np.log(original_char_prob)

        return log_total_prob

    def msa_tree_phylo(self, clade_root, method, masked, context_type, 
                        context_size, context_sampling, proposal, flip_before_start = 0, r_eff = 1.0):
                
        self.phylogeny_MSA = []

        self.sample_static_context(self.original_MSA, method = "greedy", context_size = context_size)
        first_sequence_tokens = self.mcmc(Number_of_Mutation = flip_before_start, previous_sequence_tokens = self.init_seq.clone(),
                                        method = method, masked = masked, proposal = proposal)
    
        if context_type == "static":
            self.msa_tree_phylo_recur(clade_root, first_sequence_tokens.clone(), method=method, masked=masked, proposal = proposal, r_eff = r_eff)

        elif context_type == "dynamic":
            self.msa_tree_phylo_recur_dynamic(clade_root, first_sequence_tokens.clone(), method=method, masked=masked, 
                                                context_sampling=context_sampling, context_size = context_size, proposal = proposal, r_eff = r_eff)

        results = self.phylogeny_MSA.copy()
        self.phylogeny_MSA = []
        self.context = None
        self.init_seq = None
        
        return results
    
    def msa_tree_phylo_recur(self, clade_root, previous_sequence_tokens, method, masked, proposal, r_eff):
        
        b = clade_root.clades

        if len(b)>0:
            for clade in b:
                # Mutation on previous_sequences
                # print("entering new branch")
                n_mutations = int(clade.branch_length*self.n_cols*r_eff)
                new_sequence_tokens = self.mcmc(n_mutations, previous_sequence_tokens.clone(), method, masked, proposal)
                self.msa_tree_phylo_recur(clade, new_sequence_tokens.clone(), method, masked, proposal=proposal, r_eff = r_eff)
        else:

            final_seq = ""
            for i in range(1,self.n_cols+1):

                char_index = int(previous_sequence_tokens[0,0,i].cpu().numpy())
                character = self.model_alphabet_mapping_inv[char_index]
                final_seq += character
                            
            seq_index = len(self.phylogeny_MSA)
            self.phylogeny_MSA.append((f"seq{seq_index}",final_seq))
            
    def msa_tree_phylo_recur_dynamic(self, clade_root, previous_sequence_tokens, method, masked, context_size, context_sampling, proposal, r_eff):
        
        b = clade_root.clades
        
        if len(b)>1:
            for clade in b:
                # print("entering new branch")
                n_mutations = int(clade.branch_length*self.n_cols*r_eff)
                desc_leaves = [node.name for node in clade.get_terminals() if node.name != self.start_seq_name]
                desc_sequences = [elem for elem in self.original_MSA if elem[0] in desc_leaves]
                if len(desc_leaves) > context_size:

                    if context_sampling == "random":
                        
                        random_ind = list(np.random.choice(range(len(desc_sequences)),context_size, replace = False))
                        self.context = [desc_sequences[i] for i in random_ind]

                    elif context_sampling == "greedy":
                        
                        self.context = greedy_select(desc_sequences, num_seqs = context_size)
              
                    _,_,self.context = self.batch_converter([self.context])
                    self.context = self.context.to(self.device)
                
                # new_tree = self.generate_subtree(desc_leaves)
                new_sequence_tokens = self.mcmc(n_mutations, previous_sequence_tokens.clone(), method, masked, proposal)
                self.msa_tree_phylo_recur_dynamic(clade, new_sequence_tokens.clone(), method = method, masked =masked,
                                                    context_size = context_size, context_sampling = context_sampling,
                                                    proposal = proposal, r_eff = r_eff)
        else:

            final_seq = ""
            for i in range(1,self.n_cols+1):

                char_index = int(previous_sequence_tokens[0,0,i].cpu().numpy())
                character = self.model_alphabet_mapping_inv[char_index]
                final_seq += character
                            
            seq_index = len(self.phylogeny_MSA)
            self.phylogeny_MSA.append((f"seq{seq_index}",final_seq))
   
            print(f"Number of sequences generated: {len(self.phylogeny_MSA)}")


    def msa_no_phylo(self, context_size, n_sequences, n_mutations, method, masked, proposal):

        syn_sequences_list = []

        self.sample_static_context(self.original_MSA, method = "greedy", context_size = context_size)

        for i in range(n_sequences):

            new_sequence_tokens = self.mcmc(n_mutations, self.init_seq.clone(), method, masked, proposal)

            final_seq = ""
            for i in range(1,self.n_cols+1):

                char_index = int(new_sequence_tokens[0,0,i].cpu().numpy())
                character = self.model_alphabet_mapping_inv[char_index]
                final_seq += character
                            
            seq_index = len(syn_sequences_list)
            syn_sequences_list.append((f"seq{seq_index}",final_seq))

        return syn_sequences_list
    
    def mcmc(self, Number_of_Mutation, previous_sequence_tokens, method, masked, proposal):  
    
        c_mutation = 0

        print(f"Number of mutations: {Number_of_Mutation}")
        if proposal == "random":
            proposals = 0

            proposed_probs = []
            accepted_probs = []
            acceptance_criteria = []
            accepted_acceptance_criteria = []

            previous_tokens_for_ham_dist = previous_sequence_tokens.squeeze(0).cpu().numpy()

            while c_mutation<Number_of_Mutation:

                if self.context != None:
                    stacked_tokens = torch.cat((previous_sequence_tokens, self.context), dim = 1)
                else:
                    stacked_tokens = previous_sequence_tokens.clone()
                
                selected_pos = np.random.randint(1, self.n_cols + 1)

                orig_log_prob = self.prob_calculator(stacked_tokens, selected_pos, method, masked)
                
                original_character_int = previous_sequence_tokens[0,0, selected_pos].cpu().numpy()        
                
                proposed_mutation = np.random.randint(4, 24)

                proposals += 1

                if proposed_mutation >= original_character_int:
                    proposed_mutation += 1
                
                if proposed_mutation == 24:
                    proposed_mutation = 30

                modified_sequence_tokens = previous_sequence_tokens.clone()
                modified_sequence_tokens[0,0,selected_pos] = proposed_mutation
                modified_stacked_tokens = torch.cat((modified_sequence_tokens, self.context), dim = 1)

                assert int((modified_stacked_tokens != stacked_tokens).sum().cpu().numpy()) == 1
                assert modified_stacked_tokens[0,0,selected_pos] != stacked_tokens[0,0,selected_pos]
                
                new_log_prob = self.prob_calculator(modified_stacked_tokens, selected_pos, method, masked)

                proposed_probs.append(np.exp(new_log_prob))
                
                de = new_log_prob - orig_log_prob
                acceptance_criteria.append(np.exp(de))
                    
                if (de >= 0) | (np.random.uniform() < np.exp(de)):
                    accepted_probs.append(np.exp(new_log_prob))
                    accepted_acceptance_criteria.append(np.exp(de))
                    previous_sequence_tokens = modified_sequence_tokens.clone()
                    c_mutation += 1

            if len(proposed_probs) > 0:
                mean_proposal_probs = float(np.mean(proposed_probs))
                mean_accepted_proposal_probs = float(np.mean(accepted_probs))

                mean_acceptance_criteria = float(np.mean(acceptance_criteria))
                mean_accepted_acceptance_criteria = float(np.mean(accepted_acceptance_criteria))

                new_tokens_for_hamming_dist = previous_sequence_tokens.squeeze(0).cpu().numpy()

                hamming_dist = cdist(new_tokens_for_hamming_dist,previous_tokens_for_ham_dist,"hamming").item()

                self.hamming_distances.append(hamming_dist)
                self.mean_proposal_probs.append(mean_proposal_probs)
                self.n_mutations.append(Number_of_Mutation)
                self.n_proposals.append(proposals)
                self.mean_accepted_proposal_probs.append(mean_accepted_proposal_probs)
                self.mean_acceptance_criteria.append(mean_acceptance_criteria)
                self.mean_accepted_acceptance_criteria.append(mean_accepted_acceptance_criteria)

            print(f"Number of proposals: {proposals}")

        elif proposal == "logits":

            previous_tokens_for_ham_dist = previous_sequence_tokens.squeeze(0).cpu().numpy()

            relevant_char_indices = list(range(4,24)) + [30]
            relevant_indices_mapping = {k:v for k,v in zip(relevant_char_indices,list(range(21)))}
            proposal_probs = []

            while c_mutation<Number_of_Mutation:

                if self.context != None:
                    stacked_tokens = torch.cat((previous_sequence_tokens, self.context), dim = 1)
                else:
                    stacked_tokens = previous_sequence_tokens.clone()
                
                selected_pos = np.random.randint(1, self.n_cols + 1)

                softmax = torch.nn.Softmax(dim = -1)

                stacked_tokens_copy = stacked_tokens.clone()
                original_char_index = stacked_tokens[0,0,selected_pos] 
                
                if masked == True:
                    stacked_tokens_copy[0,0,selected_pos] = self.model_alphabet_mapping["<mask>"]
                
                with torch.no_grad(): 

                    char_prob_dist = (softmax(self.model(stacked_tokens_copy)["logits"][0,0,selected_pos,:]).cpu().numpy())
                
                char_prob_dist = char_prob_dist[relevant_char_indices]
                char_prob_dist = char_prob_dist.astype('float64')
                char_prob_dist = char_prob_dist/np.sum(char_prob_dist)
                char_prob_dist = list(char_prob_dist)

                proposed_mutation = np.random.choice(relevant_char_indices, p=char_prob_dist)  
                proposed_mutation_prob = char_prob_dist[relevant_indices_mapping[proposed_mutation]] 

                if proposed_mutation == original_char_index:
                    continue 

                proposal_probs.append(proposed_mutation_prob)
                
                previous_sequence_tokens[0,0,selected_pos] = proposed_mutation
                c_mutation += 1

            if Number_of_Mutation > 0:
                new_tokens_for_hamming_dist = previous_sequence_tokens.squeeze(0).cpu().numpy()

                hamming_dist = cdist(new_tokens_for_hamming_dist,previous_tokens_for_ham_dist,"hamming").item()

                self.hamming_distances.append(hamming_dist)
                mean_proposal_probs = float(np.mean(proposal_probs))
                self.mean_proposal_probs.append(mean_proposal_probs)
                self.n_mutations.append(Number_of_Mutation)

        
        
        return previous_sequence_tokens