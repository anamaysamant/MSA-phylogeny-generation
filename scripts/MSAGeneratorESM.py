### Code credited to Viola Renner ###

from MSAGenerator import MSAGenerator
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn


class MSAGeneratorESM(MSAGenerator):
    """
    Class that generates an MSA based on a single sequence using ESM.
    """

    def __init__(self, number_of_nodes, number_state_spin, batch_size, model):
        """
        Constructor method.
        :param number_of_nodes: length of the sequence.
        :param number_state_spin: number of states (20 amino acids + 1 gap).
        :param batch_size: batch size.
        :param model: ESM model.
        """
        super().__init__(number_of_nodes, number_state_spin)
        # Set the number of batches
        self.batch_size = batch_size

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Set the model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model)
        self.model = self.model.to(self.device)

        # Set the softmax
        self.softmax_output = nn.Softmax(dim=0)

        # Set the map for computing the sequences
        self.amino_acid_map = {char: index for index, char in enumerate(self.tokenizer.all_tokens) if char in
                               ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M',
                                'H', 'W', 'C','-']}
        self.inverse_amino_acid_map = {v: k for k, v in self.amino_acid_map.items()}

    def transform_sequence_to_array(self, sequence):
        """
        Transform a string of characters in an array of integer.
        :param sequence: string of characters representing the protein sequence.
        :return: array of integer representing the protein sequence.
        """
        return np.array([self.amino_acid_map[aa] for aa in sequence if aa in self.amino_acid_map])

    def transform_array_to_sequence(self, array):
        """
        Transform an array of integer in a string of characters.
        :param array: array of integer representing the protein sequence.
        :return: string of characters representing the protein sequence.
        """
        return ''.join([self.inverse_amino_acid_map[val] for val in array])

    def msa_tree_phylo(self, clade_root, flip_before_start, first_sequence, proposal, neff=1.0):
        """
        Initialize the MSA and start the recursion to compute node sequences.
        :param clade_root: root of the tree.
        :param flip_before_start: number of mutations to apply to the random generated sequence.
        :param first_sequence: first sequence (root).
        :param neff: number of mutations per site per branch length.
        :return: MSA of the sequences in the leafs of the phylogenetic tree.
        """
        # Initialize the root
        first_sequence = self.transform_sequence_to_array(first_sequence)
        # Initialize the MSA
        msa = np.zeros((len(clade_root.get_terminals()), self.number_of_nodes), dtype=np.int8)
        # Create the new sequences recursively
        final_msa = np.asarray(self.msa_tree_phylo_recur(clade_root, first_sequence, msa, proposal, neff))
        self.cur_index = 0

        return final_msa

    def mcmc(self, number_of_mutation, l_spin, proposal):
        """
        Apply to the given sequence the given number of mutations.
        :param number_of_mutation: given number of mutations.
        :param l_spin: given sequence.
        :return: modified sequence.
        """
        # Set the number of mutations
        c_mutation = 0

        # Until the number of mutation is not achieved
        while c_mutation < number_of_mutation:
            # Select positions to mutate in the sequence

            selected_nodes = np.random.choice(np.arange(self.number_of_nodes), size=self.batch_size, replace=True)

            # Get the current sequence in string format
            protein_sequence = self.transform_array_to_sequence(l_spin)

            # Mask the positions
            masked_sequences = []
            for selected_node in selected_nodes:
                masked_sequence = protein_sequence[:selected_node] + "<mask>" + protein_sequence[selected_node + 1:]
                masked_sequences.append(masked_sequence)

             # Get the logits from the ESM model
            self.model.eval()
            with torch.no_grad():
                tokenized_sequences = self.tokenizer(masked_sequences, return_tensors="pt")
                b_input_ids = tokenized_sequences['input_ids'].to(self.device)
                b_input_mask = tokenized_sequences['attention_mask'].to(self.device)
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs[0].to('cpu')

            # Select new states

            if proposal == "random":

                new_states = np.random.randint(low = 4, high= 24, size=self.batch_size)

                # Avoid to select the same state as before
                for i in range(self.batch_size):

                    if new_states[i] >= l_spin[selected_nodes[i]]:
                        new_states[i] += 1

                    if new_states[i] == 24:
                        new_states[i] = 30

                # Evaluate if to accept the proposed mutations
                for i in range(self.batch_size):
                    # Get the probabilities of each amino acid in the masked position
                    prediction = self.softmax_output(logits[i, selected_nodes[i] + 1, :])
                    score_old = prediction[l_spin[selected_nodes[i]]]
                    score_new = prediction[new_states[i]]

                    # Compute the ratio between probabilities
                    p = score_new / score_old

                    # If the difference is positive or if it is greater than a random value, apply the mutation
                    if np.random.uniform() < p or p >= 1:
                        # Modify the selected position with the new selected state
                        l_spin[selected_nodes[i]] = new_states[i]
                        # Increase the number of mutation applied
                        c_mutation += 1
                        # Maintain only the first mutation found
                        break

            elif proposal == "logits":

                relevant_char_indices = list(range(4,24)) + [30]
                relevant_indices_mapping = {k:v for k,v in zip(relevant_char_indices,list(range(21)))}


                for i in range(self.batch_size):

                    original_char_index = l_spin[selected_nodes[i]]
                    
                    char_prob_dist = self.softmax_output(logits[i, selected_nodes[i] + 1, :]).numpy()
                    char_prob_dist = char_prob_dist[relevant_char_indices]
                    char_prob_dist = char_prob_dist.astype('float64')
                    char_prob_dist = char_prob_dist/np.sum(char_prob_dist)
                    char_prob_dist = list(char_prob_dist)

                    proposed_mutation = np.random.choice(relevant_char_indices, p=char_prob_dist)
                    proposed_mutation_prob = char_prob_dist[relevant_indices_mapping[proposed_mutation]] 

                    if proposed_mutation == original_char_index:
                        continue 
                    
                    l_spin[selected_nodes[i]] = proposed_mutation
                    c_mutation += 1

                    break

