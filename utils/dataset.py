import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

TRAIN_JSON = Path(__file__).parent.parent / "data/raw/data/standard_format/rasa/train.json"
TEST_JSON  = Path(__file__).parent.parent / "data/raw/data/standard_format/rasa/test.json"

PAD_IDX = 0
UNK_IDX = 1


def _tokenize(text: str) -> list:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def _load_examples(path: Path) -> list:
    with open(path) as f:
        return json.load(f)["rasa_nlu_data"]["common_examples"]


class Vocabulary:
    def __init__(self):
        self.token2idx = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        self.idx2token = {PAD_IDX: "<PAD>", UNK_IDX: "<UNK>"}

    def build(self, texts: list, min_freq: int = 1):
        counter = Counter()
        for text in texts:
            counter.update(_tokenize(text))
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def encode(self, text: str) -> list:
        return [self.token2idx.get(t, UNK_IDX) for t in _tokenize(text)]

    def __len__(self):
        return len(self.token2idx)


class LabelEncoder:
    def __init__(self):
        self.label2idx: dict = {}
        self.idx2label: dict = {}

    def fit(self, labels: list):
        for label in sorted(set(labels)):
            idx = len(self.label2idx)
            self.label2idx[label] = idx
            self.idx2label[idx] = label

    def encode(self, label: str) -> int:
        return self.label2idx[label]

    def __len__(self):
        return len(self.label2idx)


class ATISDataset(Dataset):
    def __init__(self, examples: list, vocab: Vocabulary, label_encoder: LabelEncoder):
        self.samples = []
        for ex in examples:
            if ex["intent"] not in label_encoder.label2idx:
                continue  # skip intents unseen during label fitting
            tokens = torch.tensor(vocab.encode(ex["text"]), dtype=torch.long)
            label  = torch.tensor(label_encoder.encode(ex["intent"]), dtype=torch.long)
            self.samples.append((tokens, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    return padded, lengths, torch.stack(labels)


def compute_class_weights(examples: list, label_encoder: LabelEncoder, max_weight: float = 10.0) -> torch.Tensor:
    counts  = Counter(ex["intent"] for ex in examples)
    n_total = sum(counts.values())
    n_cls   = len(label_encoder)
    weights = torch.zeros(n_cls)
    for label, idx in label_encoder.label2idx.items():
        weights[idx] = n_total / (n_cls * max(counts.get(label, 1), 1))
    weights = weights.clamp(max=max_weight)
    return weights


def build_dataloaders(batch_size: int = 32):
    train_examples = _load_examples(TRAIN_JSON)
    test_examples  = _load_examples(TEST_JSON)

    # vocab from train only
    vocab = Vocabulary()
    vocab.build([ex["text"] for ex in train_examples])

    # label encoder over train+test union so test encoding never crashes
    all_intents = [ex["intent"] for ex in train_examples + test_examples]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_intents)

    train_ds = ATISDataset(train_examples, vocab, label_encoder)
    test_ds  = ATISDataset(test_examples,  vocab, label_encoder)

    class_weights = compute_class_weights(train_examples, label_encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader, vocab, label_encoder, class_weights
