import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
import math


class SimpleFeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            proj_multiplier: int = 4,
            dropout: float = 0.1
        ):
        super().__init__()
        self.d_model = d_model
        self.proj_multiplier = proj_multiplier
        self.dropout = dropout

        self.up_proj = nn.Linear(self.d_model, self.d_model * self.proj_multiplier, bias=False)
        self.down_proj = nn.Linear(self.d_model * self.proj_multiplier, self.d_model, bias=False)
        self.dropout = nn.Dropout(self.dropout)
        self.init_weights()
    
    def init_weights(self):
        std = self.d_model ** -0.5
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std)
    
    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.up_proj(hidden_states)
        hidden_states = F.relu(hidden_states).square()
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

        