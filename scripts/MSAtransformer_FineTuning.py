import esm
import torch
from time import time

import numpy as np
from Bio import SeqIO, Phylo
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from torch.utils.data import Dataset, DataLoader
from aux_msa_functions import remove_insertions
from select_gpu import get_free_gpu

torch.set_grad_enabled(True)
import torch.nn as nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu())

import logging

logging.basicConfig(
    filename='fine_tuning.log',               
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode="a"
)

class MSADataset(Dataset):
    def __init__(self, masked_tokens, true_tokens):
        self.masked_tokens = masked_tokens
        self.true_tokens = true_tokens


    def __len__(self):
        return self.masked_tokens.shape[0]

    def __getitem__(self, idx):

        masked_MSA, true_MSA = self.masked_tokens[idx], self.true_tokens[idx]

        return masked_MSA, true_MSA

def create_training_set(all_seqs, seqs_per_MSA, n_sampled_MSAs, p_mask, mask_idx, seed):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    masked_MSAs = []
    true_MSAs = []
    
    np.random.seed(seed)
    
    for i in range(n_sampled_MSAs):
    
        sampled_ids = list(np.random.choice(range(len(all_seqs)), seqs_per_MSA, replace=False))
        sampled_MSA = [all_seqs[i] for i in sampled_ids]
    
        _,_,batch_tokens = batch_converter([sampled_MSA])
    
        starting_tokens = batch_tokens[0,:,:1]
        batch_tokens = batch_tokens[0,:,1:]
    
        mask = ((torch.rand(batch_tokens.shape) > p_mask).type(
                    torch.uint8))
    
        masked_batch_tokens = batch_tokens * mask + mask_idx * (1 - mask)
    
        batch_tokens = torch.cat((starting_tokens, batch_tokens), dim = -1)
        masked_batch_tokens = torch.cat((starting_tokens, masked_batch_tokens), dim = -1)
    
        masked_MSAs.append(masked_batch_tokens)
        true_MSAs.append(batch_tokens)
    
    masked_MSAs = torch.stack(masked_MSAs, dim=0).to(device)
    true_MSAs = torch.stack(true_MSAs, dim=0).to(device)
    
    return masked_MSAs, true_MSAs

# Hyper-parameters 
num_epochs = 10
batch_size = 1
learning_rate = 1e-4
weight_decay = 1e-4
# lr_scheduler: str = "warmup_linear"
# warmup_steps: int = 16000
adam_betas = (0.9, 0.999)
max_steps: int = 1000000

seqs_per_MSA = 300
n_sampled_MSAs = 1000
p_mask = 0.1

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
batch_converter = alphabet.get_batch_converter()
mask_idx = alphabet.tok_to_idx["<mask>"]

for name, param in model.named_parameters():
    if not name.startswith("lm_head"):
        param.requires_grad = False

model.lm_head.weight.requires_grad = True

FT_Fam = "PF00004"

if os.path.exists(f"MSA_finetuned_{FT_Fam}.pt"):
    model = torch.load(f"MSA_finetuned_{FT_Fam}.pt")

model = model.to(device)


MSA_filename = f"../data/protein-families-msa-full/{FT_Fam}_full.fasta"
all_seqs = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(MSA_filename, "fasta")]

masked_MSAs, true_MSAs = create_training_set(all_seqs=all_seqs, seqs_per_MSA=seqs_per_MSA, n_sampled_MSAs= n_sampled_MSAs, 
                                             p_mask=p_mask, mask_idx=mask_idx, seed=42)

MSAs_Dataset = MSADataset(masked_MSAs,true_MSAs)
train_dataloader = DataLoader(MSAs_Dataset, batch_size = batch_size, shuffle = True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=adam_betas)  

# Train the model
n_total_steps = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (masked_MSAs, true_MSAs) in enumerate(train_dataloader):  

        logits = model(masked_MSAs)["logits"]
        masked_pos = masked_MSAs == alphabet.tok_to_idx["<mask>"]

        logits = logits[masked_pos].to(device)
        true_MSAs = true_MSAs[masked_pos].to(device)

        loss = criterion(logits, true_MSAs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        del masked_MSAs, true_MSAs, logits

torch.save(model,f"MSA_finetuned_{FT_Fam}.pt")
