# Intent Classification for Natural Language Queries

**EE 541: A Computational Introduction to Deep Learning — Final Project**  
Zhipeng Huang | University of Southern California

---

## Overview

This project implements and compares two deep learning models for intent classification on the ATIS (Airline Travel Information System) dataset — a multi-class text classification task with 26 intent categories and significant class imbalance.

| Model | Accuracy | Macro F1 |
|---|---|---|
| TextCNN (baseline) | 95.07% | 0.636 |
| BiLSTM + Attention | 96.53% | 0.692 |

---

## Repository Structure

```
EE541-project/
├── data/
│   └── raw/                        # ATIS dataset (cloned from GitHub)
│       └── data/standard_format/rasa/
│           ├── train.json          # 4,978 training examples
│           └── test.json           # 893 test examples
├── models/
│   ├── textcnn.py                  # TextCNN baseline model
│   └── bilstm_attention.py         # BiLSTM + Attention primary model
├── utils/
│   └── dataset.py                  # Data loading, vocab, label encoding, DataLoader
├── checkpoints/                    # Saved model weights (auto-created on training)
│   ├── textcnn_best.pt
│   └── bilstm_best.pt
├── config.py                       # All hyperparameters in one place
├── train.py                        # Training loop (shared by both models)
├── evaluate.py                     # Test-set evaluation, per-class F1, confusion matrix
```

> **Note:** Do not commit `checkpoints/` or `data/raw/` to the repository per submission guidelines.

---

## Setup

### Requirements

- Python 3.9+
- PyTorch 2.10.0
- scikit-learn
- matplotlib

### Installation

```bash
# 1. Clone this repository
git clone <your-repo-url>
cd EE541-project

# 2. Install dependencies
pip install torch scikit-learn matplotlib

# 3. Download the ATIS dataset
git clone https://github.com/howl-anderson/ATIS_dataset data/raw
```

No GPU is required. The code automatically detects and uses MPS (Apple Silicon), CUDA, or CPU.

---

## How to Run

### Training

```bash
# Train TextCNN baseline (seed=42 used for reported results)
python train.py --model textcnn --seed 42

# Train BiLSTM + Attention
python train.py --model bilstm --seed 42
```

Training prints a per-epoch table of train/test loss and accuracy. The best checkpoint (by test accuracy) is automatically saved to `checkpoints/<model>_best.pt`.

### Evaluation

```bash
# Evaluate TextCNN on the test set
python evaluate.py --model textcnn

# Evaluate BiLSTM + Attention
python evaluate.py --model bilstm
```

Outputs a full `classification_report` (per-class precision, recall, F1) and saves a confusion matrix image to `checkpoints/<model>_confusion.png`.


---

## Data Format

The dataset is loaded from Rasa JSON format. Each example has the structure:

```json
{
  "text": "i want to fly from boston at 838 am and arrive in denver",
  "intent": "flight",
  "entities": [...]
}
```

Only `text` and `intent` fields are used (slot/entity labels are ignored).

**Preprocessing pipeline:**
1. Lowercase + remove non-alphanumeric characters
2. Word-level tokenization
3. Vocabulary built from training set only (`<PAD>=0`, `<UNK>=1`)
4. Label encoder fit on train + test union (so test-only classes don't cause crashes)
5. Dynamic padding per batch via `collate_fn` — no fixed max length

---

## Hyperparameter Configuration

All hyperparameters are in `config.py`:

```python
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3        # Adam, with ReduceLROnPlateau (patience=3, factor=0.5)
EMBED_DIM   = 128
DROPOUT     = 0.5

# TextCNN
CNN_FILTERS      = 128
CNN_KERNEL_SIZES = (2, 3, 4, 5)

# BiLSTM
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
```

**Reported results use `--seed 42`.**

---

## Reproducibility

To reproduce the reported results exactly:

```bash
python train.py --model textcnn --seed 42
python evaluate.py --model textcnn

python train.py --model bilstm --seed 42
python evaluate.py --model bilstm
```

Seeds are fixed for Python `random`, NumPy, and PyTorch at the start of each training run. Note: exact bit-for-bit reproduction may vary across hardware (MPS vs CUDA vs CPU) due to floating-point non-determinism in parallel operations.

---

## Model Architectures

### TextCNN

```
Embedding(888, 128)
→ 4× parallel Conv1d (kernel=2,3,4,5, filters=128 each)
→ Max-over-time pooling per kernel
→ Concat → (512,)
→ Dropout(0.5) → Linear(512, 26)
```
Total parameters: **356,890**

### BiLSTM + Attention

```
Embedding(888, 128)
→ Dropout(0.5)
→ BiLSTM(hidden=128, layers=2) with pack/unpack for variable-length sequences
→ Additive attention: score = tanh(W·h), padding masked to −∞
→ Weighted sum → context vector (256,)
→ Dropout(0.5) → Linear(256, 26)
```
Total parameters: **845,850**

---

## Class Imbalance Handling

The ATIS dataset is heavily skewed (`flight` = 73.6% of training data). This is addressed via **Weighted CrossEntropyLoss**, where each class weight is computed as:

```
weight[c] = total_samples / (num_classes × count[c])
```

Weights are capped at 10.0 to prevent instability from extremely rare classes.
