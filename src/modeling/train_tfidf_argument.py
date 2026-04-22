import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_split, build_label_indexer, print_eval_report, Trainer
from tfidf_classifier import tokenize, build_vocab, compute_tfidf, SMT_LABELS, NON_SMT_LABELS
from train_argument_features import extract_argument_features, FEATURE_NAMES


DATA_DIR = Path(__file__).parent.parent.parent / "data"

NUM_EPOCHS = 70
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.0005
HIDDEN_SIZE = 96
DROPOUT = 0.6


def main():
    torch.manual_seed(50)
    np.random.seed(50)

    # load data
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

    # tfidf features
    train_tokens = [tokenize(t) for t in train_texts]
    dev_tokens = [tokenize(t) for t in dev_texts]
    test_tokens = [tokenize(t) for t in test_texts]

    vocab = build_vocab(train_tokens)
    print(f"TF-IDF vocab size: {len(vocab)}")

    train_tfidf, idf = compute_tfidf(train_tokens, vocab)
    dev_tfidf, _ = compute_tfidf(dev_tokens, vocab, idf=idf)
    test_tfidf, _ = compute_tfidf(test_tokens, vocab, idf=idf)

    # argument structure features
    print(f"Extracting {len(FEATURE_NAMES)} argument structure features...")
    train_arg = np.array([extract_argument_features(t) for t in train_texts])
    dev_arg = np.array([extract_argument_features(t) for t in dev_texts])
    test_arg = np.array([extract_argument_features(t) for t in test_texts])

    mu = train_arg.mean(axis=0)
    sigma = train_arg.std(axis=0)
    sigma[sigma == 0] = 1.0
    train_arg = (train_arg - mu) / sigma
    dev_arg = (dev_arg - mu) / sigma
    test_arg = (test_arg - mu) / sigma

    train_vecs = np.hstack([train_tfidf, train_arg])
    dev_vecs = np.hstack([dev_tfidf, dev_arg])
    test_vecs = np.hstack([test_tfidf, test_arg])
    input_dim = train_vecs.shape[1]

    print(
        f"Combined feature vector: {input_dim} "
        f"(tfidf={train_tfidf.shape[1]} + argument={len(FEATURE_NAMES)})"
    )

    # labels
    label_indexer = build_label_indexer(train_labels)
    num_classes = len(label_indexer)
    label_names = [label_indexer.get_object(i) for i in range(num_classes)]

    train_y = torch.tensor(
        [label_indexer.index_of(lbl) for lbl in train_labels], dtype=torch.long
    )
    dev_y = torch.tensor(
        [label_indexer.index_of(lbl) for lbl in dev_labels], dtype=torch.long
    )
    test_y = torch.tensor(
        [label_indexer.index_of(lbl) for lbl in test_labels], dtype=torch.long
    )

    # data loaders
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(train_vecs), train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    dev_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(dev_vecs), dev_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(test_vecs), test_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # hidden layer
    model = nn.Sequential(
        nn.Linear(input_dim, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_SIZE, num_classes),
    )

    # Getting loss weights
    counts = Counter(train_y.tolist())
    n_samples = len(train_y)
    w = torch.tensor(
        [n_samples / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
    )
    loss_fn = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # train and evaluate
    trainer = Trainer(
        model, train_loader, dev_loader, loss_fn, optimizer, num_epochs=NUM_EPOCHS
    )
    trainer.train()

    print("\n--- Dev set ---")
    targets, preds = trainer.pred()
    print_eval_report(targets, preds, label_names)

    print("\n--- Test set ---")
    model.eval()
    test_targets, test_preds = [], []
    with torch.no_grad():
        for x, y in test_loader:
            test_preds.extend(model(x).argmax(-1).tolist())
            test_targets.extend(y.tolist())
    print_eval_report(test_targets, test_preds, label_names)

    # Confusion matrix of F1
    labels = list(range(num_classes))
    cm = confusion_matrix(test_targets, test_preds, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("TF-IDF + Argument Features Confusion Matrix")
    for i in range(num_classes):
        for j in range(num_classes):
            v = cm_norm[i, j]
            if v > 0.01:
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=8,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

if __name__ == "__main__":
    main()
