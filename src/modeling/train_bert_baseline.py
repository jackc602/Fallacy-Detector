import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from utils import Indexer, print_eval_report

path = Path(__file__).parent.parent.parent / "data"

MODEL_NAME    = "roberta-base"
NUM_EPOCHS    = 20
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
MAX_LENGTH    = 128


def load_split(split):
    with open(path / f"{split}_fol_clean_gemini_gemini-2.5-pro.json", encoding="utf-8") as f:
        return json.load(f)


class FallacyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def build_label_indexer(labels):
    indexer = Indexer()
    for lbl in sorted(set(labels)):
        indexer.add_and_get_index(lbl)
    return indexer



def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    train_data = load_split("train")
    dev_data   = load_split("dev")
    test_data  = load_split("test")

    train_texts  = [item["text"] for item in train_data]
    dev_texts    = [item["text"] for item in dev_data]
    test_texts   = [item["text"] for item in test_data]
    train_labels = [item["label"] for item in train_data]
    dev_labels   = [item["label"] for item in dev_data]
    test_labels  = [item["label"] for item in test_data]

    # labels
    label_indexer = build_label_indexer(train_labels)
    num_classes   = len(label_indexer)
    label_names   = [label_indexer.get_object(i) for i in range(num_classes)]

    train_y = torch.tensor([label_indexer.index_of(lbl) for lbl in train_labels], dtype=torch.long)
    dev_y   = torch.tensor([label_indexer.index_of(lbl) for lbl in dev_labels], dtype=torch.long)
    test_y  = torch.tensor([label_indexer.index_of(lbl) for lbl in test_labels], dtype=torch.long)

    print(f"Train: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}, Classes: {num_classes}")

    # tokenize with pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes
    ).to(device)

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    dev_enc   = tokenizer(dev_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    test_enc  = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")

    train_loader = DataLoader(FallacyDataset(train_enc, train_y), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(FallacyDataset(dev_enc, dev_y), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(FallacyDataset(test_enc, test_y), batch_size=BATCH_SIZE, shuffle=False)

    # class-weighted loss
    counts = Counter(train_y.tolist())
    n_samples = len(train_y)
    w = torch.tensor(
        [n_samples / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # training loop
    best_dev_acc = 0.0
    best_state   = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, total = 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            loss = loss_fn(out.logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch["labels"].size(0)
            total      += batch["labels"].size(0)

        # dev evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in dev_loader:
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device))
                preds = out.logits.argmax(dim=-1)
                correct += (preds == batch["labels"].to(device)).sum().item()

        dev_acc = correct / len(dev_y)
        print(f"Epoch {epoch}  train_loss={total_loss/total:.4f}  dev_acc={dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest dev accuracy: {best_dev_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    label_names = [label_indexer.get_object(i) for i in range(num_classes)]

    # collect predictions and evaluate
    model.eval()

    print("\n--- Dev set ---")
    targets, preds = [], []
    with torch.no_grad():
        for batch in dev_loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            targets.extend(batch["labels"].tolist())
    print_eval_report(targets, preds, label_names)

    print("\n--- Test set ---")
    targets, preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            targets.extend(batch["labels"].tolist())
    print_eval_report(targets, preds, label_names)


if __name__ == "__main__":
    main()
