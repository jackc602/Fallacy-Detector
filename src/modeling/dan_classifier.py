import torch
import torch.nn as nn
import json
from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from utils import Indexer, WordEmbeddings, print_eval_report


DATA_DIR = Path(__file__).parent.parent.parent / "data"

NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.00001
HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 2
DROPOUT = 0.4
FREEZE_EMBEDDINGS = True

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# formal fallacies
FORMAL_LABELS = [
    "false dilemma",
    "circular reasoning",
    "fallacy of logic",
    "false causality",
    "faulty generalization",
    "equivocation",
]

# informal fallacies
INFORMAL_LABELS = [
    "ad hominem",
    "ad populum",
    "appeal to emotion",
    "fallacy of credibility",
    "fallacy of extension",
    "fallacy of relevance",
    "intentional",
]


def load_split_labels(split):
    with open(
        DATA_DIR / f"{split}_fol_clean_gemini_gemini-2.5-pro.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
    return [item["label"] for item in data]


def load_split_tokens(split):
    with open(DATA_DIR / f"{split}_nl_tokens.json", "r") as f:
        return json.load(f)


def load_word_embeddings():
    with open(DATA_DIR / "word_vocab.json", "r") as f:
        vocab = json.load(f)
    vectors = np.load(DATA_DIR / "word_embeddings.npy")

    indexer = Indexer()
    ordered = sorted(vocab.items(), key=lambda kv: kv[1])
    for word, idx in ordered:
        assigned = indexer.add_and_get_index(word)
        assert assigned == idx, f"vocab/index mismatch for {word}"

    vec_list = list(vectors)
    return WordEmbeddings(indexer, vec_list), vocab


def build_label_indexer(labels):
    indexer = Indexer()
    for label in sorted(set(labels)):
        indexer.add_and_get_index(label)
    return indexer


def encode_labels(labels, indexer):
    return [indexer.index_of(lbl) for lbl in labels]


class DANDataset(Dataset):
    def __init__(self, token_lists, label_ids):
        self.examples = list(zip(token_lists, label_ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate(pad_idx, unk_idx):
    """
    Custom collate funciton for out dataloader to conver embeddings to tensors
    and apply pad token.
    """

    def collate(batch):
        token_lists, labels = zip(*batch)
        token_lists = [t if len(t) > 0 else [unk_idx] for t in token_lists]
        max_len = max(len(t) for t in token_lists)
        x = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        for i, t in enumerate(token_lists):
            x[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            lengths[i] = len(t)
        y = torch.tensor(labels, dtype=torch.long)
        return x, lengths, y

    return collate


class DAN(nn.Module):
    def __init__(
        self,
        word_embeddings,
        num_classes,
        pad_idx,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        dropout=DROPOUT,
        freeze_embeddings=FREEZE_EMBEDDINGS,
    ):
        super().__init__()
        # initialize embeddings
        self.pad_idx = pad_idx
        self.embedding = word_embeddings.get_initialized_embedding_layer(
            frozen=freeze_embeddings, padding_idx=pad_idx
        )
        embed_dim = word_embeddings.get_embedding_length()

        # define feedforward architecture
        layers = [nn.Dropout(dropout)]
        prev = embed_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, token_ids, lengths):
        embs = self.embedding(token_ids)
        # get the indices of non pad tokens
        mask = (token_ids != self.pad_idx).unsqueeze(-1).float()
        summed = (embs * mask).sum(dim=1)
        # ensure we divide by number of actual (non pad) tokens
        denom = lengths.clamp(min=1).unsqueeze(-1).float()
        avg = summed / denom
        return self.classifier(avg)


def run_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total = 0
    for x, lengths, y in loader:
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets, all_preds = [], []
    for x, lengths, y in loader:
        logits = model(x, lengths)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=-1)
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
    for x, lengths, y in loader:
        preds.extend(model(x, lengths).argmax(dim=-1).tolist())
        targets.extend(y.tolist())
    return targets, preds


def main():
    torch.manual_seed(50)
    np.random.seed(50)

    word_embeddings, vocab = load_word_embeddings()
    pad_idx = word_embeddings.word_indexer.index_of(PAD_TOKEN)
    unk_idx = word_embeddings.word_indexer.index_of(UNK_TOKEN)

    # load in data
    train_tokens = load_split_tokens("train")
    dev_tokens = load_split_tokens("dev")
    test_tokens = load_split_tokens("test")

    # load labels and load them into indexers
    train_labels = load_split_labels("train")
    dev_labels = load_split_labels("dev")
    test_labels = load_split_labels("test")

    # # logical fallacies only
    # pairs = [(t, l) for t, l in zip(train_tokens, train_labels) if l in FORMAL_LABELS]
    # train_tokens, train_labels = zip(*pairs)
    # pairs = [(t, l) for t, l in zip(dev_tokens, dev_labels) if l in FORMAL_LABELS]
    # dev_tokens, dev_labels = zip(*pairs)
    # pairs = [(t, l) for t, l in zip(test_tokens, test_labels) if l in FORMAL_LABELS]
    # test_tokens, test_labels = zip(*pairs)

    # # informal fallacies only (only one uncommented at a time)
    # pairs = [(t, l) for t, l in zip(train_tokens, train_labels) if l in INFORMAL_LABELS]
    # train_tokens, train_labels = zip(*pairs)
    # pairs = [(t, l) for t, l in zip(dev_tokens, dev_labels) if l in INFORMAL_LABELS]
    # dev_tokens, dev_labels = zip(*pairs)
    # pairs = [(t, l) for t, l in zip(test_tokens, test_labels) if l in INFORMAL_LABELS]
    # test_tokens, test_labels = zip(*pairs)

    label_indexer = build_label_indexer(train_labels)
    num_classes = len(label_indexer)

    train_y = encode_labels(train_labels, label_indexer)
    dev_y = encode_labels(dev_labels, label_indexer)
    test_y = encode_labels(test_labels, label_indexer)

    # package data instances and wrap in data loader object with our custom collate function
    train_ds = DANDataset(train_tokens, train_y)
    dev_ds = DANDataset(dev_tokens, dev_y)
    test_ds = DANDataset(test_tokens, test_y)

    collate = make_collate(pad_idx, unk_idx)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    model = DAN(
        word_embeddings=word_embeddings, num_classes=num_classes, pad_idx=pad_idx
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    counts = Counter(train_y)
    class_weights = torch.tensor(
        [len(train_y) / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_dev_f1 = -1.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn)
        dev_loss, dev_acc, dev_f1 = evaluate(model, dev_loader, loss_fn)
        print(
            f"Epoch {epoch}, train_loss={train_loss:.4f}, "
            f"dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}, "
            f"dev_macro_f1={dev_f1:.4f}"
        )
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    print(f"Best dev macro F1: {best_dev_f1}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # aggregate predictions and label names and run in depth eval
    label_names = [label_indexer.get_object(i) for i in range(num_classes)]

    print("\n--- Dev set ---")
    targets, preds = collect_predictions(model, dev_loader)
    print_eval_report(targets, preds, label_names)

    print("\n--- Test set ---")
    targets, preds = collect_predictions(model, test_loader)
    print_eval_report(targets, preds, label_names)

    out_path = DATA_DIR.parent / "src" / "models" / "dan_best.pt"
    torch.save(
        {"model_state": best_state, "label_indexer": label_indexer.objs_to_ints},
        out_path,
    )


if __name__ == "__main__":
    main()
