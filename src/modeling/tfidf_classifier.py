import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from utils import build_label_indexer, print_eval_report


DATA_DIR = Path(__file__).parent.parent.parent / "data"

NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MIN_DF = 2

SMT_LABELS = [
    "false dilemma",
    "circular reasoning",
    "fallacy of logic",
    "false causality",
    "faulty generalization",
    "equivocation",
]

NON_SMT_LABELS = [
    "ad hominem",
    "ad populum",
    "appeal to emotion",
    "fallacy of credibility",
    "fallacy of extension",
    "fallacy of relevance",
    "intentional",
]


def load_split(split):
    with open(
        DATA_DIR / f"{split}_fol_clean_gemini_gemini-2.5-pro.json", encoding="utf-8"
    ) as f:
        return json.load(f)


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def build_vocab(token_lists, min_df=MIN_DF):
    df = defaultdict(int)
    for tokens in token_lists:
        for tok in set(tokens):
            df[tok] += 1
    vocab = {}
    idx = 0
    for tok, count in sorted(df.items()):
        if count >= min_df:
            vocab[tok] = idx
            idx += 1
    return vocab


def compute_tfidf(token_lists, vocab, idf=None):
    N = len(token_lists)
    V = len(vocab)
    tf_matrix = np.zeros((N, V), dtype=np.float32)

    for i, tokens in enumerate(token_lists):
        vocab_tokens = [t for t in tokens if t in vocab]
        counts = Counter(vocab_tokens)
        total = max(len(tokens), 1)
        for tok, cnt in counts.items():
            tf_matrix[i, vocab[tok]] = cnt / total

    if idf is None:
        df = (tf_matrix > 0).sum(axis=0).astype(np.float32)
        idf = np.log((N + 1) / (df + 1)) + 1.0

    tfidf = tf_matrix * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf /= norms

    return tfidf, idf


class TFIDFDataset(Dataset):
    def __init__(self, vectors, label_ids):
        self.X = torch.tensor(vectors, dtype=torch.float32)
        self.y = torch.tensor(label_ids, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def run_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, total = 0.0, 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_targets, all_preds = [], []
    for x, y in loader:
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * y.size(0)
        preds = logits.argmax(-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_targets.extend(y.tolist())
        all_preds.extend(preds.tolist())
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / total, correct / total, macro_f1


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    targets, preds = [], []
    for x, y in loader:
        preds.extend(model(x).argmax(-1).tolist())
        targets.extend(y.tolist())
    return targets, preds


def main():
    torch.manual_seed(50)
    np.random.seed(50)

    train_data = load_split("train")
    dev_data = load_split("dev")
    test_data = load_split("test")

    train_texts = [item["text"] for item in train_data]
    dev_texts = [item["text"] for item in dev_data]
    test_texts = [item["text"] for item in test_data]
    train_labels = [item["label"] for item in train_data]
    dev_labels = [item["label"] for item in dev_data]
    test_labels = [item["label"] for item in test_data]

    # # logical fallacies only
    # pairs = [(t, l) for t, l in zip(train_texts, train_labels) if l in SMT_LABELS]
    # train_texts, train_labels = zip(*pairs)
    # pairs = [(t, l) for t, l in zip(dev_texts, dev_labels) if l in SMT_LABELS]
    # dev_texts, dev_labels = zip(*pairs)
    # pairs = [(t, l) for t, l in zip(test_texts, test_labels) if l in SMT_LABELS]
    # test_texts, test_labels = zip(*pairs)

    # # informal fallacies only (swap with block above)
    # pairs = [(t, l) for t, l in zip(train_texts, train_labels) if l in NON_SMT_LABELS]
    # train_texts, train_labels = zip(*pairs)
    # pairs = [(t, l) for t, l in zip(dev_texts, dev_labels) if l in NON_SMT_LABELS]
    # dev_texts, dev_labels = zip(*pairs)
    # pairs = [(t, l) for t, l in zip(test_texts, test_labels) if l in NON_SMT_LABELS]
    # test_texts, test_labels = zip(*pairs)

    train_tokens = [tokenize(t) for t in train_texts]
    dev_tokens = [tokenize(t) for t in dev_texts]
    test_tokens = [tokenize(t) for t in test_texts]

    vocab = build_vocab(train_tokens)
    print(f"Vocab size: {len(vocab)}")

    train_vecs, idf = compute_tfidf(train_tokens, vocab)
    dev_vecs, _ = compute_tfidf(dev_tokens, vocab, idf=idf)
    test_vecs, _ = compute_tfidf(test_tokens, vocab, idf=idf)

    label_indexer = build_label_indexer(train_labels)
    num_classes = len(label_indexer)
    label_names = [label_indexer.get_object(i) for i in range(num_classes)]

    train_y = [label_indexer.index_of(l) for l in train_labels]
    dev_y = [label_indexer.index_of(l) for l in dev_labels]
    test_y = [label_indexer.index_of(l) for l in test_labels]

    train_loader = DataLoader(
        TFIDFDataset(train_vecs, train_y), batch_size=BATCH_SIZE, shuffle=True
    )
    dev_loader = DataLoader(
        TFIDFDataset(dev_vecs, dev_y), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TFIDFDataset(test_vecs, test_y), batch_size=BATCH_SIZE, shuffle=False
    )

    # logistic regression - single linear layer
    model = nn.Linear(len(vocab), num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    counts = Counter(train_y)
    class_weights = torch.tensor(
        [len(train_y) / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_dev_f1, best_state = -1.0, None
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn)
        dev_loss, dev_acc, dev_f1 = evaluate(model, dev_loader, loss_fn)
        print(
            f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.4f}  "
            f"dev_macro_f1={dev_f1:.4f}"
        )
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest dev macro F1: {best_dev_f1:.4f}")
    model.load_state_dict(best_state)

    print("\n--- Dev set ---")
    targets, preds = collect_predictions(model, dev_loader)
    print_eval_report(targets, preds, label_names)

    print("\n--- Test set ---")
    targets, preds = collect_predictions(model, test_loader)
    print_eval_report(targets, preds, label_names)


if __name__ == "__main__":
    main()
