from dataclasses import dataclass

import torch
from torch import nn

@dataclass
class TransformerConfig:
    ...

class Embeddings(nn.Module):
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