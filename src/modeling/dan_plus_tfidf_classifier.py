import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from utils import Indexer, WordEmbeddings, print_eval_report


DATA_DIR = Path(__file__).parent.parent.parent / "data"

NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.00001
HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 2
TFIDF_PROJ_DIM = 128
DROPOUT = 0.4
FREEZE_EMBEDDINGS = True
MIN_DF = 2

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

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


def load_split_raw(split):
    with open(
        DATA_DIR / f"{split}_fol_clean_gemini_gemini-2.5-pro.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


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

    vec_list = list(vectors)
    return WordEmbeddings(indexer, vec_list), vocab


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def build_tfidf_vocab(token_lists, min_df=MIN_DF):
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


def build_label_indexer(labels):
    indexer = Indexer()
    for lbl in sorted(set(labels)):
        indexer.add_and_get_index(lbl)
    return indexer


def encode_labels(labels, indexer):
    return [indexer.index_of(lbl) for lbl in labels]


class HybridDataset(Dataset):
    def __init__(self, token_lists, tfidf_matrix, label_ids):
        assert len(token_lists) == len(tfidf_matrix) == len(label_ids)
        self.token_lists = list(token_lists)
        self.tfidf = torch.tensor(tfidf_matrix, dtype=torch.float32)
        self.labels = torch.tensor(label_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_lists[idx], self.tfidf[idx], self.labels[idx]


def make_collate(pad_idx, unk_idx):
    def collate(batch):
        token_lists, tfidf_rows, labels = zip(*batch)
        token_lists = [t if len(t) > 0 else [unk_idx] for t in token_lists]
        max_len = max(len(t) for t in token_lists)
        x = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        for i, t in enumerate(token_lists):
            x[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            lengths[i] = len(t)
        tfidf = torch.stack(tfidf_rows, dim=0)
        y = torch.stack(labels, dim=0)
        return x, lengths, tfidf, y

    return collate


class HybridClassifier(nn.Module):
    def __init__(
        self,
        word_embeddings,
        tfidf_vocab_size,
        num_classes,
        pad_idx,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        tfidf_proj_dim=TFIDF_PROJ_DIM,
        dropout=DROPOUT,
        freeze_embeddings=FREEZE_EMBEDDINGS,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = word_embeddings.get_initialized_embedding_layer(
            frozen=freeze_embeddings, padding_idx=pad_idx
        )
        embed_dim = word_embeddings.get_embedding_length()

        # learned bottleneck so the (much larger) tf-idf stream doesn't swamp
        # the averaged embedding by dimensionality alone
        self.tfidf_proj = nn.Sequential(
            nn.Linear(tfidf_vocab_size, tfidf_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        layers = [nn.Dropout(dropout)]
        prev = embed_dim + tfidf_proj_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, token_ids, lengths, tfidf):
        embs = self.embedding(token_ids)
        mask = (token_ids != self.pad_idx).unsqueeze(-1).float()
        summed = (embs * mask).sum(dim=1)
        denom = lengths.clamp(min=1).unsqueeze(-1).float()
        avg = summed / denom

        tproj = self.tfidf_proj(tfidf)
        combined = torch.cat([avg, tproj], dim=-1)
        return self.classifier(combined)


def run_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, total = 0.0, 0
    for x, lengths, tfidf, y in loader:
        optimizer.zero_grad()
        logits = model(x, lengths, tfidf)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, lengths, tfidf, y in loader:
        logits = model(x, lengths, tfidf)
        total_loss += loss_fn(logits, y).item() * y.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    targets, preds = [], []
    for x, lengths, tfidf, y in loader:
        preds.extend(model(x, lengths, tfidf).argmax(-1).tolist())
        targets.extend(y.tolist())
    return targets, preds


def main():
    torch.manual_seed(50)
    np.random.seed(50)

    word_embeddings, _ = load_word_embeddings()
    pad_idx = word_embeddings.word_indexer.index_of(PAD_TOKEN)
    unk_idx = word_embeddings.word_indexer.index_of(UNK_TOKEN)

    train_raw = load_split_raw("train")
    dev_raw = load_split_raw("dev")
    test_raw = load_split_raw("test")

    train_token_ids = load_split_tokens("train")
    dev_token_ids = load_split_tokens("dev")
    test_token_ids = load_split_tokens("test")

    train_texts = [item["text"] for item in train_raw]
    dev_texts = [item["text"] for item in dev_raw]
    test_texts = [item["text"] for item in test_raw]
    train_labels = [item["label"] for item in train_raw]
    dev_labels = [item["label"] for item in dev_raw]
    test_labels = [item["label"] for item in test_raw]

    ## logical fallacies only
    # triples = [(tok, txt, lbl) for tok, txt, lbl in zip(train_token_ids, train_texts, train_labels) if lbl in SMT_LABELS]
    # train_token_ids, train_texts, train_labels = [list(x) for x in zip(*triples)]
    # triples = [(tok, txt, lbl) for tok, txt, lbl in zip(dev_token_ids, dev_texts, dev_labels) if lbl in SMT_LABELS]
    # dev_token_ids, dev_texts, dev_labels = [list(x) for x in zip(*triples)]
    # triples = [(tok, txt, lbl) for tok, txt, lbl in zip(test_token_ids, test_texts, test_labels) if lbl in SMT_LABELS]
    # test_token_ids, test_texts, test_labels = [list(x) for x in zip(*triples)]

    ## informal fallacies only (swap with block above)
    # triples = [(tok, txt, lbl) for tok, txt, lbl in zip(train_token_ids, train_texts, train_labels) if lbl in NON_SMT_LABELS]
    # train_token_ids, train_texts, train_labels = [list(x) for x in zip(*triples)]
    # triples = [(tok, txt, lbl) for tok, txt, lbl in zip(dev_token_ids, dev_texts, dev_labels) if lbl in NON_SMT_LABELS]
    # dev_token_ids, dev_texts, dev_labels = [list(x) for x in zip(*triples)]
    # triples = [(tok, txt, lbl) for tok, txt, lbl in zip(test_token_ids, test_texts, test_labels) if lbl in NON_SMT_LABELS]
    # test_token_ids, test_texts, test_labels = [list(x) for x in zip(*triples)]

    # tf-idf stream uses its own regex tokenizer over raw text; the pretrained
    # embedding vocab is separate and drives the dan stream via the token-id files
    train_text_tokens = [tokenize(t) for t in train_texts]
    dev_text_tokens = [tokenize(t) for t in dev_texts]
    test_text_tokens = [tokenize(t) for t in test_texts]

    tfidf_vocab = build_tfidf_vocab(train_text_tokens)
    print(f"TF-IDF vocab size: {len(tfidf_vocab)}")

    train_tfidf, idf = compute_tfidf(train_text_tokens, tfidf_vocab)
    dev_tfidf, _ = compute_tfidf(dev_text_tokens, tfidf_vocab, idf=idf)
    test_tfidf, _ = compute_tfidf(test_text_tokens, tfidf_vocab, idf=idf)

    label_indexer = build_label_indexer(train_labels)
    num_classes = len(label_indexer)
    label_names = [label_indexer.get_object(i) for i in range(num_classes)]

    train_y = encode_labels(train_labels, label_indexer)
    dev_y = encode_labels(dev_labels, label_indexer)
    test_y = encode_labels(test_labels, label_indexer)

    train_ds = HybridDataset(train_token_ids, train_tfidf, train_y)
    dev_ds = HybridDataset(dev_token_ids, dev_tfidf, dev_y)
    test_ds = HybridDataset(test_token_ids, test_tfidf, test_y)

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

    model = HybridClassifier(
        word_embeddings=word_embeddings,
        tfidf_vocab_size=len(tfidf_vocab),
        num_classes=num_classes,
        pad_idx=pad_idx,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()

    best_dev_acc, best_state = 0.0, None
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn)
        dev_loss, dev_acc = evaluate(model, dev_loader, loss_fn)
        print(
            f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
            f"dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.4f}"
        )
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    print(f"Best dev accuracy: {best_dev_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n--- Dev set ---")
    targets, preds = collect_predictions(model, dev_loader)
    print_eval_report(targets, preds, label_names)

    print("\n--- Test set ---")
    targets, preds = collect_predictions(model, test_loader)
    print_eval_report(targets, preds, label_names)

    out_path = DATA_DIR.parent / "src" / "models" / "dan_tfidf_best.pt"
    torch.save(
        {"model_state": best_state, "label_indexer": label_indexer.objs_to_ints},
        out_path,
    )


if __name__ == "__main__":
    main()
