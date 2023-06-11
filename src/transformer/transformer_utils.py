import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class TransformerConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    vocab_size: int = 30522

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
    def __init__(self, config: TransformerConfig) -> None:
        super(MultiHeadAttention, self).__init__()
        num_heads = config.num_attention_heads
        embedding_dim = config.hidden_size # Keeping HuggingFace notation
        head_dim = embedding_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embedding_dim, head_dim) for _ in range(num_heads)])
        self.linear = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x, attention_mask) for head in self.heads], dim=-1)
        return self.linear(out)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out # Logits

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.layer_norm1(x)
        x += self.attention(out, attention_mask)
        x += self.feed_forward(self.layer_norm2(x))

        return x

class TransfomerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super(TransfomerEncoder, self).__init__()
        self.embedding = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x