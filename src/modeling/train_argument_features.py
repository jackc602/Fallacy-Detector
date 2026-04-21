import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_split,
    build_label_indexer,
    print_eval_report,
    Trainer,
)


DATA_DIR = Path(__file__).parent.parent.parent / "data"

NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


## Lists of words to use as features, these were examples generated with gemini

PREMISE_MARKERS = [
    "because",
    "since",
    "given",
    "as",
    "for",
    "due",
    "owing",
    "whereas",
    "considering",
    "assuming",
    "granted",
]
CONCLUSION_MARKERS = [
    "therefore",
    "thus",
    "hence",
    "so",
    "consequently",
    "accordingly",
    "implies",
    "means",
    "proves",
    "shows",
]
NEGATIONS = [
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "nobody",
    "nothing",
    "nowhere",
]
POSITIVE_WORDS = [
    "love",
    "happy",
    "joy",
    "wonderful",
    "beautiful",
    "hope",
    "great",
    "amazing",
    "fantastic",
    "excellent",
    "good",
    "best",
    "proud",
    "grateful",
    "blessed",
]
NEGATIVE_WORDS = [
    "hate",
    "fear",
    "afraid",
    "terrible",
    "horrible",
    "evil",
    "sad",
    "angry",
    "awful",
    "disgusting",
    "worst",
    "bad",
    "dangerous",
    "threat",
    "pain",
    "suffer",
    "tragic",
]


## Feature names that we are adding

FEATURE_NAMES = [
    "premise_density",
    "conclusion_density",
    "negation_density",
    "positive_density",
    "negative_density",
    "sentiment_polarity",
    "question_per_sentence",
    "exclamation_per_sentence",
    "capitalization_ratio",
    "type_token_ratio",
    "text_length",
    "avg_word_length",
    "avg_sentence_length",
    "sentence_count",
]


## feature extraction


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def word_density(words, word_set, n):
    count = len([w for w in words if w in word_set])
    return count / n


def extract_argument_features(text):
    words = tokenize(text)
    n = max(len(words), 1)

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    num_sentences = max(len(sentences), 1)
    pos = word_density(words, POSITIVE_WORDS, n)
    neg = word_density(words, NEGATIVE_WORDS, n)

    features = [
        word_density(words, PREMISE_MARKERS, n),
        word_density(words, CONCLUSION_MARKERS, n),
        word_density(words, NEGATIONS, n),
        pos,
        neg,
        pos - neg,
        text.count("?") / num_sentences,
        text.count("!") / num_sentences,
        len([c for c in text if c.isupper()]) / max(len(text), 1),
        len(set(words)) / n,
        n / 100.0,
        sum(len(w) for w in words) / len(words) if words else 0.0,
        n / num_sentences / 20.0,
        num_sentences / 10.0,
    ]

    features = np.array(features, dtype=np.float32)
    return features


def print_feature_importance(model, feature_names):
    W = model[0].weight.data.abs().mean(dim=0).numpy()
    ranked = sorted(zip(feature_names, W), key=lambda x: x[1], reverse=True)
    print("\nFeatures by avg absolute weight:")
    for name, weight in ranked:
        print(f"  {name:25s} {weight:.4f}")


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

    # extract and normalize features
    num_features = len(FEATURE_NAMES)
    print(f"Extracting {num_features} features...")
    train_features = np.array([extract_argument_features(t) for t in train_texts])
    dev_features = np.array([extract_argument_features(t) for t in dev_texts])
    test_features = np.array([extract_argument_features(t) for t in test_texts])

    mu = train_features.mean(axis=0)
    sigma = train_features.std(axis=0)
    sigma[sigma == 0] = 1.0
    train_features = (train_features - mu) / sigma
    dev_features = (dev_features - mu) / sigma
    test_features = (test_features - mu) / sigma

    print(f"Feature matrix shape: {train_features.shape}")

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
        torch.utils.data.TensorDataset(torch.tensor(train_features), train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    dev_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(dev_features), dev_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(test_features), test_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # using feedforward network
    model = nn.Sequential(
        nn.Linear(num_features, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes),
    )

    # class-weighted loss
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

    # train
    trainer = Trainer(
        model, train_loader, dev_loader, loss_fn, optimizer, num_epochs=NUM_EPOCHS
    )
    trainer.train()

    # evaluation
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
    print_feature_importance(model, FEATURE_NAMES)


if __name__ == "__main__":
    main()
