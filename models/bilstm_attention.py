import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128,
                 hidden_dim=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # attention: 可学习的 query 向量，对每个时间步打分
        self.attn_w = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_q = nn.Linear(hidden_dim * 2, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        # x: (batch, seq_len)
        emb = self.dropout(self.embedding(x))           # (batch, seq_len, embed_dim)

        # pack → lstm → unpack，让 LSTM 忽略 padding 位置
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)  # (batch, seq_len, hidden*2)

        # attention
        # 先对每个时间步算一个 score，再把 padding 位置 mask 掉
        score = self.attn_q(torch.tanh(self.attn_w(output)))       # (batch, seq_len, 1)
        score = score.squeeze(-1)                                    # (batch, seq_len)

        # mask: padding 位置设为 -inf，softmax 后权重趋近 0
        max_len = output.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        score = score.masked_fill(mask, float("-inf"))

        weights = F.softmax(score, dim=1)                           # (batch, seq_len)
        context = (weights.unsqueeze(-1) * output).sum(dim=1)      # (batch, hidden*2)

        return self.fc(self.dropout(context))                       # (batch, num_classes)
