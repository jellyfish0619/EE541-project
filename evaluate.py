import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config
from utils.dataset import build_dataloaders


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for padded, lengths, labels in loader:
        padded, lengths = padded.to(device), lengths.to(device)
        logits = model(padded, lengths)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def run(model_name: str):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    _, test_loader, vocab, label_encoder, _ = build_dataloaders(config.BATCH_SIZE)

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

    ckpt_path = config.CKPT_DIR / f"{model_name}_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    labels, preds = predict(model, test_loader, device)

    present_ids = sorted(set(labels) | set(preds))
    class_names  = [label_encoder.idx2label[i] for i in present_ids]

    # per-class report
    print(f"\n=== {model_name.upper()} — Classification Report ===\n")
    print(classification_report(labels, preds, labels=present_ids,
                                target_names=class_names, digits=3, zero_division=0))

    # overall
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"Overall accuracy: {acc:.4f}")

    # confusion matrix
    cm = confusion_matrix(labels, preds, labels=present_ids)
    fig = plot_confusion_matrix(cm, class_names, f"{model_name} — Confusion Matrix")
    out_path = config.CKPT_DIR / f"{model_name}_confusion.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nConfusion matrix saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bilstm", choices=["textcnn", "bilstm"])
    args = parser.parse_args()
    run(args.model)

