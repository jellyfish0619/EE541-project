import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128,
                 num_filters=128, kernel_sizes=(2, 3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len)
        x = self.embedding(x)               # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)             # (batch, embed_dim, seq_len)  Conv1d 要求这个顺序

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))             # (batch, num_filters, seq_len - k + 1)
            h = h.max(dim=2).values         # (batch, num_filters)
            pooled.append(h)

        x = torch.cat(pooled, dim=1)        # (batch, num_filters * 4)
        x = self.dropout(x)
        return self.fc(x)                   # (batch, num_classes)
