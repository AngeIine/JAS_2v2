from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalGraphLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        out = torch.bmm(adj, support)
        out = torch.relu(out)
        out = self.norm(out)
        return self.dropout(out)


class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        q = self.q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_mask = mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(attn_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.out(attended), weights


class MultiTaskTemporalGCN(nn.Module):
    def __init__(
        self,
        num_numeric_features: int,
        num_actions: int,
        num_positions: int,
        num_sports: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        embed_dim = hidden_dim // 4
        self.action_embed = nn.Embedding(num_actions, embed_dim)
        self.position_embed = nn.Embedding(num_positions, embed_dim)
        self.sport_embed = nn.Embedding(num_sports, embed_dim)
        self.numeric_proj = nn.Linear(num_numeric_features, hidden_dim - embed_dim * 3)

        self.gcn1 = TemporalGraphLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.gcn2 = TemporalGraphLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.attn = MultiHeadTemporalAttention(hidden_dim, num_heads=num_heads, dropout=dropout)

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def build_temporal_adj(self, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        adj = torch.zeros((batch_size, seq_len, seq_len), device=mask.device)
        for i in range(seq_len):
            adj[:, i, i] = 1.0
            if i > 0:
                adj[:, i, i - 1] = 1.0
        adj = adj * mask.unsqueeze(1) * mask.unsqueeze(2)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return adj / degree

    def forward(self, batch: dict) -> dict:
        numeric = batch["numeric"]
        action_hist = batch["action_history"]
        position_hist = batch["position_history"]
        sport_hist = batch["sport_history"]
        mask = batch["mask"]

        x = torch.cat(
            [
                self.numeric_proj(numeric),
                self.action_embed(action_hist),
                self.position_embed(position_hist),
                self.sport_embed(sport_hist),
            ],
            dim=-1,
        )

        adj = self.build_temporal_adj(mask)
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        x_attn, attn_weights = self.attn(x, mask)
        x = x + x_attn

        last_index = mask.sum(dim=1).long() - 1
        batch_index = torch.arange(x.size(0), device=x.device)
        pooled = x[batch_index, last_index]

        return {
            "score_logits": self.score_head(pooled),
            "action_logits": self.action_head(pooled),
            "attention_weights": attn_weights,
        }
