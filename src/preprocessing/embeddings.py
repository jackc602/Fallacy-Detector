import re
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

data_path = Path(__file__).parent.parent.parent / "data"

DEFAULT_MODEL = "all-MiniLM-L6-v2"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")



def tokenize(text: str):
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def load_split_token_lists(in_path):
    """
    Load in a file and tokenize it
    """
    with open(in_path, "r") as f:
        data = json.load(f)
    return [tokenize(item["text"]) for item in data]


def build_vocab(token_lists_by_split):
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token_lists in token_lists_by_split:
        for tokens in token_lists:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


def encode_vocab(vocab, model_name=DEFAULT_MODEL):
    """
    Call our embedding model to get vecs
    """
    model = SentenceTransformer(model_name)
    words = [w for w, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    vectors = model.encode(words, show_progress_bar=True, batch_size=256)
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors[vocab[PAD_TOKEN]] = 0.0
    return vectors


def tokens_to_ids(token_lists, vocab):
    unk = vocab[UNK_TOKEN]
    return [[vocab.get(t, unk) for t in toks] for toks in token_lists]


def main():
    datasets = [
        ("train_fol_clean.json", "train"),
        ("test_fol_clean.json", "test"),
        ("dev_fol_clean.json", "dev"),
    ]

    nl_tokens_by_split = {}
    fol_tokens_by_split = {}
    for filename, split in datasets:
        in_path = data_path / filename
        nl_tokens_by_split[split] = load_split_token_lists(in_path)
        fol_tokens_by_split[split] = load_split_token_lists(in_path)

    # pull vocab out of all words in all splits, then embed and save
    vocab = build_vocab(list(nl_tokens_by_split.values()) + list(fol_tokens_by_split.values()))
    vectors = encode_vocab(vocab)
    np.save(data_path / "word_embeddings.npy", vectors)
    print(f"saved embeddings {data_path / "word_embeddings.npy"}")
    with open(data_path / "word_vocab.json", "w") as f:
        json.dump(vocab, f)
    print(f"saved vocab json at {data_path / "word_vocab.json"}")

    # Save token ids of the tokens in each sentence so we can directly go 
    # from 
    for split, toks in nl_tokens_by_split.items():
        ids = tokens_to_ids(toks, vocab)
        with open(data_path / f"{split}_nl_tokens.json", "w") as f:
            json.dump(ids, f)
    for split, toks in fol_tokens_by_split.items():
        ids = tokens_to_ids(toks, vocab)
        with open(data_path / f"{split}_fol_tokens.json", "w") as f:
            json.dump(ids, f)



if __name__ == "__main__":
    main()
