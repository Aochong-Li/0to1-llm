import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import SimpleAttentionBlock
from mlp import SimpleFeedForward

class GPTConfig:
    max_seq_len: int = 4096
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 6
    d_model: int = 768
    proj_multiplier: int = 4
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx

        self.attention = SimpleAttentionBlock(
            d_model=self.config.d_model,
            num_heads=self.config.n_heads,
            dropout=self.config.dropout,
            max_seq_len=self.config.max_seq_len,
            bias=False
            )
        self.mlp = SimpleFeedForward(
            d_model=self.config.d_model,
            proj_multiplier=self.config.proj_multiplier,
            dropout=self.config.dropout
        )
        self.layer_norm = nn.LayerNorm(self.config.d_model)
    
    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.layer_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm(hidden_states))
        
        return hidden_states