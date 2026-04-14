import re
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

data_path = Path(__file__).parent.parent.parent / "data"

DEFAULT_MODEL = "all-MiniLM-L6-v2"

FOL_SYMBOL_REPLACEMENTS = {
    "∀": " for all ",
    "∃": " there exists ",
    "→": " implies ",
    "↔": " if and only if ",
    "∧": " and ",
    "∨": " or ",
    "¬": " not ",
    "∈": " is in ",
    "≈": " approximately equals ",
    "≠": " not equals ",
    "≤": " less than or equal to ",
    "≥": " greater than or equal to ",
}

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WS_RE = re.compile(r"\s+")


def _split_camel_case(token: str) -> str:
    return _CAMEL_RE.sub(" ", token)


def fol_to_natural_language(fol: str) -> str:
    if not fol:
        return ""

    text = fol
    for symbol, phrase in FOL_SYMBOL_REPLACEMENTS.items():
        text = text.replace(symbol, phrase)

    text = " ".join(_split_camel_case(tok) for tok in text.split())

    # remove added whitespace
    text = _WS_RE.sub(" ", text).strip()
    return text


def embed_file_text(in_path, out_path, model_name=DEFAULT_MODEL):
    model = SentenceTransformer(model_name)

    with open(in_path, "r") as f:
        data = json.load(f)
    print(f"Loaded data from {in_path}")

    texts = [item["text"] for item in data]
    embeddings = model.encode(texts, show_progress_bar=True)

    np.save(out_path, embeddings)
    print(f"Saved embeddings to {out_path}")


def embed_file_fol(in_path, out_path, model_name=DEFAULT_MODEL):
    model = SentenceTransformer(model_name)

    with open(in_path, "r") as f:
        data = json.load(f)
    print(f"Loaded data from {in_path}")

    texts = [fol_to_natural_language(item.get("fol", "")) for item in data]
    embeddings = model.encode(texts, show_progress_bar=True)

    np.save(out_path, embeddings)
    print(f"Saved embeddings to {out_path}")


def main():
    datasets = [
        ("train_fol_clean_gemini_gemini-2.5-pro.json", "train"),
        ("test_fol_clean_gemini_gemini-2.5-pro.json", "test"),
        ("dev_fol_clean_gemini_gemini-2.5-pro.json", "dev"),
    ]

    for filename, split in datasets:
        in_path = data_path / filename
        embed_file_text(in_path, data_path / f"{split}_nl_embeddings.npy")
        embed_file_fol(in_path, data_path / f"{split}_fol_embeddings.npy")


if __name__ == "__main__":
    main()
