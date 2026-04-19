import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_split, build_label_indexer, print_eval_report, Trainer
from tfidf_classifier import tokenize, build_vocab, compute_tfidf
from train_argument_features import extract_argument_features, FEATURE_NAMES


DATA_DIR = Path(__file__).parent.parent.parent / "data"

NUM_EPOCHS    = 200
BATCH_SIZE    = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY  = 0.0001


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # load data
    train_data = load_split("train")
    dev_data   = load_split("dev")

    train_texts  = [item["text"] for item in train_data]
    dev_texts    = [item["text"] for item in dev_data]
    train_labels = [item["label"] for item in train_data]
    dev_labels   = [item["label"] for item in dev_data]

    # tfidf features
    train_tokens = [tokenize(t) for t in train_texts]
    dev_tokens   = [tokenize(t) for t in dev_texts]

    vocab = build_vocab(train_tokens)
    print(f"TF-IDF vocab size: {len(vocab)}")

    train_tfidf, idf = compute_tfidf(train_tokens, vocab)
    dev_tfidf,   _   = compute_tfidf(dev_tokens, vocab, idf=idf)

    # argument structure features
    print(f"Extracting {len(FEATURE_NAMES)} argument structure features...")
    train_arg = np.array([extract_argument_features(t) for t in train_texts])
    dev_arg   = np.array([extract_argument_features(t) for t in dev_texts])

    mu    = train_arg.mean(axis=0)
    sigma = train_arg.std(axis=0)
    sigma[sigma == 0] = 1.0
    train_arg = (train_arg - mu) / sigma
    dev_arg   = (dev_arg - mu) / sigma

    train_vecs = np.hstack([train_tfidf, train_arg])
    dev_vecs   = np.hstack([dev_tfidf, dev_arg])
    input_dim  = train_vecs.shape[1]

    print(f"Combined feature vector: {input_dim} "
          f"(tfidf={train_tfidf.shape[1]} + argument={len(FEATURE_NAMES)})")

    # labels
    label_indexer = build_label_indexer(train_labels)
    num_classes   = len(label_indexer)
    label_names   = [label_indexer.get_object(i) for i in range(num_classes)]

    train_y = torch.tensor([label_indexer.index_of(lbl) for lbl in train_labels], dtype=torch.long)
    dev_y   = torch.tensor([label_indexer.index_of(lbl) for lbl in dev_labels], dtype=torch.long)

    # data loaders
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(train_vecs), train_y),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    dev_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(dev_vecs), dev_y),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # hidden layer
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, num_classes),
    )

    # Getting loss weights 
    counts    = Counter(train_y.tolist())
    n_samples = len(train_y)
    w = torch.tensor(
        [n_samples / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
    )
    loss_fn   = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # train and evaluate
    trainer = Trainer(model, train_loader, dev_loader, loss_fn, optimizer, num_epochs=NUM_EPOCHS)
    trainer.train()

    targets, preds = trainer.collect_predictions()
    print_eval_report(targets, preds, label_names)


if __name__ == "__main__":
    main()
