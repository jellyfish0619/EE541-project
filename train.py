import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from utils.dataset import build_dataloaders


def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for padded, lengths, labels in loader:
        padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(padded, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for padded, lengths, labels in loader:
        padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
        logits = model(padded, lengths)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


def run(model_name: str, seed: int = 42):
    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}  |  Seed: {seed}")

    train_loader, test_loader, vocab, label_encoder, class_weights = \
        build_dataloaders(config.BATCH_SIZE)

    if model_name == "textcnn":
        from models.textcnn import TextCNN
        model = TextCNN(
            vocab_size=len(vocab),
            num_classes=len(label_encoder),
            embed_dim=config.EMBED_DIM,
            num_filters=config.CNN_FILTERS,
            kernel_sizes=config.CNN_KERNEL_SIZES,
            dropout=config.DROPOUT,
        )
    elif model_name == "bilstm":
        from models.bilstm_attention import BiLSTMAttention
        model = BiLSTMAttention(
            vocab_size=len(vocab),
            num_classes=len(label_encoder),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            dropout=config.DROPOUT,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = Adam(model.parameters(), lr=config.LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_acc, best_epoch = 0.0, 0
    ckpt_path = config.CKPT_DIR / f"{model_name}_best.pt"

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Test Loss':>9}  {'Test Acc':>8}")
    print("-" * 55)

    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = eval_epoch(model, test_loader,  criterion, device)
        scheduler.step(te_loss)

        marker = "  *" if te_acc > best_acc else ""
        if te_acc > best_acc:
            best_acc, best_epoch = te_acc, epoch
            torch.save(model.state_dict(), ckpt_path)

        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  {te_loss:>9.4f}  {te_acc:>8.4f}{marker}")

    print(f"\nBest test acc: {best_acc:.4f}  (epoch {best_epoch})")
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="textcnn", choices=["textcnn", "bilstm"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.model, args.seed)
