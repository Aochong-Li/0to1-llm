import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
import math

class SimpleAttentionBlock(nn.Module):
    """
    The original implementation of the attention block from the paper "Attention is all you need" https://arxiv.org/pdf/1706.03762 and GPT-2 https://arxiv.org/abs/2205.11916
    """
    def __init__(self,
                d_model: int,
                num_heads: int,
                dropout: float = 0.1,
                max_seq_len: int = 4096,
                bias: bool = False
                ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads
        self.max_seq_len = max_seq_len

        causal_mask = torch.triu(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1)
        causal_mask = torch.where(causal_mask == 0., causal_mask, -torch.inf)
        self.register_buffer("causal_mask", causal_mask)
        self.bias = bias
        
        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.d_head, self.bias)
        self.k_proj = nn.Linear(self.d_model, self.num_heads * self.d_head, self.bias)
        self.v_proj = nn.Linear(self.d_model, self.num_heads * self.d_head, self.bias)
        self.out_proj = nn.Linear(self.num_heads * self.d_head, self.d_model, self.bias)
        self.attn_dropout = nn.Dropout(dropout)

        assert self.num_heads * self.d_head == self.d_model, "d_model is not divisible by num_heads"
        
        self.init_weights()

    def init_weights(self):
        std = self.d_head ** -0.5

        nn.init.normal_(self.q_proj.weight, mean=0.0, std = std)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std = std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std = std)

        nn.init.normal_(self.out_proj.weight, mean=0.0, std = std)

    def forward(self, hidden_states: torch.Tensor):
        """
        For a input matrix X of shape (L, D), and W_Q, W_K, W_V matrices
        Step 1. Get Q, K, V matrices for each head
        Step 2. For each head, compute softmax(QK^T / sqrt(d_k)) V
        Step 3. cConcatenate all the heads and multiply with W_O
        """
        batch_size, seq_len, d_model = hidden_states.size()
        assert d_model == self.d_model, "The hidden dimension does not match"
        assert seq_len <= self.max_seq_len, "The sequence length exceeds max_seq_len"
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, d_head)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, d_head)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, d_head)

        attention_score = torch.matmul(q, k.transpose(2,3)) / math.sqrt(self.d_head)
        attention_score = attention_score + self.causal_mask[:seq_len, :seq_len]
        attention_score = F.softmax(attention_score, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        attention_score = self.attn_dropout(attention_score)

        outputs = torch.matmul(attention_score, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.num_heads * self.d_head)
        print(outputs.size())

        outputs = self.out_proj(outputs)

        return outputs
