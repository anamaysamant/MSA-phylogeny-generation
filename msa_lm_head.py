from typing import Optional, Tuple, List, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field

from tokenization import Vocab

from modules import (
    TransformerLayer,
    AxialTransformerLayer,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    RowSelfAttention,
    ColumnSelfAttention,
)

@dataclass
class TransformerLayerConfig:
    embed_dim: int = 768
    num_attention_heads: int = 12
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    attention_type: str = "standard"
    performer_attention_features: int = 256


@dataclass
class PKMLayerConfig(TransformerLayerConfig):
    pkm_attention_heads: int = 8
    num_product_keys: int = 1024
    pkm_topk: int = 32


@dataclass
class TransformerConfig:
    layer: TransformerLayerConfig = TransformerLayerConfig()
    pkm: PKMLayerConfig = PKMLayerConfig()
    num_layers: int = 12
    max_seqlen: int = 1024
    pkm_layers: List[int] = field(default_factory=list)


@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "warmup_linear"
    warmup_steps: int = 16000
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    max_steps: int = 1000000


class BaseProteinModel(pl.LightningModule, ABC):
    def __init__(
        self,
        vocab: Vocab,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
    ):
        super().__init__()
        self.vocab = vocab
        self.optimizer_config = optimizer_config

    @abstractmethod
    def forward(
        self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        return NotImplemented

    @abstractmethod
    def get_sequence_attention(self, tokens):
        return NotImplemented

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    
class MSATransformer_lm_head(BaseProteinModel):
    def __init__(
        self,
        vocab: Vocab,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        embed_dim: int = 768,
    ):
        super().__init__(
            vocab=vocab,
            optimizer_config=optimizer_config,
        )

        self.embed_tokens = nn.Embedding(
            len(vocab), embed_dim, padding_idx=vocab.pad_idx
        )

        self.lm_head = RobertaLMHead(
            embed_dim=embed_dim,
            output_dim=len(self.vocab),
            weight=self.embed_tokens.weight,
        )

        self.init_weights()

    def forward(
        self, x
    ):

        x = self.lm_head(x)

        return x

    def max_tokens_per_msa_(self, value: int) -> None:
        """The MSA Transformer automatically batches attention computations when
        gradients are disabled to allow you to pass in larger MSAs at test time than
        you can fit in GPU memory. By default this occurs when more than 2^14 tokens
        are passed in the input MSA. You can set this value to infinity to disable
        this behavior.
        """
        self.max_tokens_per_msa = value
        for module in self.modules():
            if isinstance(module, (RowSelfAttention, ColumnSelfAttention)):
                module.max_tokens_per_msa = value

    def get_sequence_attention(self, tokens):
        return self(tokens.to(device=self.device), need_head_weights=True)[
            "row_attentions"
        ]

