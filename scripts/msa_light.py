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

    
class MSATransformer(BaseProteinModel):
    def __init__(
        self,
        vocab: Vocab,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        embed_dim: int = 768,
        num_attention_heads: int = 12,
        num_layers: int = 12,
        embed_positions_msa: bool = True,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2 ** 14,
        max_seqlen: int = 1024,
    ):
        super().__init__(
            vocab=vocab,
            optimizer_config=optimizer_config,
        )
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.embed_positions_msa = embed_positions_msa
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_tokens_per_msa = max_tokens_per_msa

        self.embed_tokens = nn.Embedding(
            len(vocab), embed_dim, padding_idx=vocab.pad_idx
        )

        if embed_positions_msa:
            self.msa_position_embedding = nn.Parameter(
                0.01 * torch.randn(1, 1024, 1, embed_dim),
                requires_grad=True,
            )
        else:
            self.register_parameter("msa_position_embedding", None)  # type: ignore

        self.dropout_module = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    embedding_dim=embed_dim,
                    ffn_embedding_dim=4 * embed_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    max_tokens_per_msa=max_tokens_per_msa,
                )
                for _ in range(num_layers)
            ]
        )

        # self.contact_head = ContactPredictionHead(
        #     num_layers * num_attention_heads,
        #     vocab.prepend_bos,
        #     vocab.append_eos,
        #     eos_idx=vocab.eos_idx,
        # )
        # self.contact_head.requires_grad_(False)
        self.embed_positions = LearnedPositionalEmbedding(
            max_seqlen,
            embed_dim,
            vocab.pad_idx,
        )
        self.emb_layer_norm_before = nn.LayerNorm(embed_dim)
        self.emb_layer_norm_after = nn.LayerNorm(embed_dim)
        # self.lm_head = RobertaLMHead(
        #     embed_dim=embed_dim,
        #     output_dim=len(self.vocab),
        #     weight=self.embed_tokens.weight,
        # )

        self.init_weights()

    def forward(
        self, tokens, need_head_weights=False
    ):
        # if return_contacts:
        #     need_head_weights = True

        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.vocab.pad_idx)  # B, R, C
        if not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(tokens)
        x += self.embed_positions(
            tokens.view(batch_size * num_alignments, seqlen)
        ).view(x.size())
        if self.msa_position_embedding is not None:
            if x.size(1) > 1024:
                raise RuntimeError(
                    "Using model with MSA position embedding trained on maximum MSA "
                    f"depth of 1024, but received {x.size(1)} alignments."
                )
            x += self.msa_position_embedding[:, :num_alignments]

        x = self.emb_layer_norm_before(x)

        x = self.dropout_module(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # repr_layers = set(repr_layers)
        # hidden_representations = {}
        # if 0 in repr_layers:
        #     hidden_representations[0] = x

        # if need_head_weights:
        #     row_attn_weights = []
        #     col_attn_weights = []

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            # if need_head_weights:
            #     x, col_attn, row_attn = x
            #     # H x C x B x R x R -> B x H x C x R x R
            #     col_attn_weights.append(col_attn.permute(2, 0, 1, 3, 4))
            #     # H x B x C x C -> B x H x C x C
            #     row_attn_weights.append(row_attn.permute(1, 0, 2, 3))
            # if (layer_idx + 1) in repr_layers:
            #     hidden_representations[layer_idx + 1] = x.permute(2, 0, 1, 3)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D

        # last hidden representation should have layer norm applied
        # if (layer_idx + 1) in repr_layers:
        #     hidden_representations[layer_idx + 1] = x

        # x = self.lm_head(x)

        # result = {"representations": hidden_representations}
        # if need_head_weights:
        #     # col_attentions: B x L x H x C x R x R
        #     col_attentions = torch.stack(col_attn_weights, 1)
        #     # row_attentions: B x L x H x C x C
        #     row_attentions = torch.stack(row_attn_weights, 1)
        #     result["col_attentions"] = col_attentions
        #     result["row_attentions"] = row_attentions
        #     if return_contacts:
        #         contacts = self.contact_head(tokens, row_attentions)
        #         result["contacts"] = contacts

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

    # @classmethod
    # def from_esm(cls):
    #     import esm
    #     from evo.tokenization import Vocab

    #     esm_model, alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
    #     args = esm_model.args
    #     vocab = Vocab.from_esm_alphabet(alphabet)
    #     model = cls(
    #         vocab=vocab,
    #         embed_dim=args.embed_dim,
    #         num_attention_heads=args.attention_heads,
    #         num_layers=args.layers,
    #         embed_positions_msa=args.embed_positions_msa,
    #         dropout=args.dropout,
    #         attention_dropout=args.attention_dropout,
    #         activation_dropout=args.activation_dropout,
    #         max_tokens_per_msa=getattr(args, "max_tokens_per_msa", args.max_tokens),
    #     )

    #     model.load_state_dict(esm_model.state_dict())

    #     return model
