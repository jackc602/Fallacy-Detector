import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn
from sklearn.metrics import classification_report, confusion_matrix

class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.ints_to_objs = {}
        self.objs_to_ints = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]
    
class WordEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_initialized_embedding_layer(self, frozen=True, padding_idx=None):
        """
        :param frozen: True if you want the embedding layer to stay frozen, false to fine-tune embeddings
        :param padding_idx: Set to a value that you want to be labeled as "padding" in the embedding space
        :return: torch.nn.Embedding layer you can use in your network
        """
        return torch.nn.Embedding.from_pretrained(torch.FloatTensor(np.array(self.vectors)),
                                                  freeze=frozen, padding_idx=padding_idx)

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]


# Data loading and eval
path = Path(__file__).parent.parent.parent / "data"


def load_split(split):
    with open(path / f"{split}_fol_clean_gemini_gemini-2.5-pro.json", encoding="utf-8") as f:
        return json.load(f)


def build_label_indexer(labels):
    indexer = Indexer()
    for label in sorted(set(labels)):
        indexer.add_and_get_index(label)
    return indexer


def print_eval_report(targets, preds, label_names):
    labels = list(range(len(label_names)))
    cm = confusion_matrix(targets, preds, labels=labels)
    print("\nConfusion matrix (rows = true, cols = pred):")
    print(pd.DataFrame(cm, index=label_names, columns=label_names).to_string())
    print("\nClassification report:")
    print(classification_report(targets, preds, labels=labels, target_names=label_names, digits=3, zero_division=0))


class Trainer:
    def __init__(self, model, train_loader, dev_loader, loss_fn,
                 optimizer, num_epochs=200):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def run_epoch(self):
        """One epoch of training"""
        self.model.train()
        tloss = 0
        batches = 0
        for x, y in self.train_loader:
            self.optimizer.zero_grad()
            scores = self.model(x)
            loss = self.loss_fn(scores, y)
            loss.backward()
            self.optimizer.step()
            tloss += loss.item()
            batches += 1
        return tloss / batches

    def evaluate(self):
        """Evaluating on dev, returns average loss and accuracy"""
        self.model.eval()
        tloss = 0
        correct = 0
        total = 0
        batches = 0
        with torch.no_grad():
            for x, y in self.dev_loader:
                scores = self.model(x)
                loss = self.loss_fn(scores, y)
                preds = torch.argmax(scores, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                tloss += loss.item()
                batches += 1
        return tloss / batches, correct / total

    def pred(self):
        """Gets predictions"""
        self.model.eval()
        targets, preds = [], []
        with torch.no_grad():
            for x, y in self.dev_loader:
                scores = self.model(x)
                preds.extend(torch.argmax(scores, dim=1).tolist())
                targets.extend(y.tolist())
        return targets, preds

    def train(self):
        """Trains model and returns best dev accuracy"""
        best_acc = 0.0
        best_state = None

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.run_epoch()
            dev_loss, dev_acc = self.evaluate()
            print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.4f}")

            if dev_acc > best_acc:
                best_acc = dev_acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}

        print(f"\nBest dev accuracy: {best_acc:.4f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_acc