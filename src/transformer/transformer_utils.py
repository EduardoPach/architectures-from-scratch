import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class TransformerConfig:
    ...

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
    """Computes scaled dot product attention for a batch of queries, keys and values.
    If mask is provided, masked positions are not included when computing attention.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor of shape (batch_size, seq_len, head_dim)
    key : torch.Tensor
        Key tensor of shape (batch_size, seq_len, head_dim)
    value : torch.Tensor
        Value tensor of shape (batch_size, seq_len, head_dim)
    mask : torch.Tensor | None, optional
        Mask tensor to apply to attention scores, by default None

    Returns
    -------
    torch.Tensor
        Updated value tensor of shape (batch_size, seq_len, head_dim)
    """
    d = query.size(-1)
    attention_score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -float('inf'))
    weights = F.softmax(attention_score, dim=-1)
    return torch.bmm(weights, value)

class Embeddings(nn.Module):
    """Combines token and position embeddings.

    Parameters
    ----------
    config : TransformerConfig
        Transformer configuration
    """
    def __init__(self, config: TransformerConfig) -> None:
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0) # Adding batch dimension
        # Getting embeddings
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(position_ids)
        # Summing embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    
class AttentionHead(nn.Module):
    def __init__(self, embedding_dim: int, head_dim: int) -> None:
        super(AttentionHead, self).__init__()
        self.query = nn.Linear(embedding_dim, head_dim)
        self.key = nn.Linear(embedding_dim, head_dim)
        self.value = nn.Linear(embedding_dim, head_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention(
            self.query(x),
            self.key(x),
            self.value(x),
            attention_mask
        )

class MultiHeadAttention(nn.Module):
    ...

class FeedForward(nn.Module):
    ...

class TransformerEncoderLayer(nn.Module):
    ...

class TransfomerEncoder(nn.Module):
    ...