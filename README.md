# Fallacy-Detector

We sought to classify short textual arguments into one of 13 logical
fallacy types, and also into two sub-tasks. These were formal (logical structure based) and informal (content based) fallacy types.

## Approaches

Several models are trained and compared (see `notes.md` for full per-class
results):

- **TF-IDF + Logistic Regression** — baseline.
- **DAN** — deep averaging network over pretrained word embeddings.
- **DAN + TF-IDF hybrid** — concatenates averaged embeddings with a projected
  TF-IDF vector.
- **Argument-structure features** — 14 handcrafted features (premise /
  conclusion density, sentiment, pronouns, etc).
- **TF-IDF + argument features** — lexical + structural features.
- **BERT / RoBERTa** — transformer fine-tuning.


## Setup

Package management is done using uv, with details listed in 'pyproject.toml'. Best practice would to have uv installed and run 'uv sync' after cloning.

```bash
git clone https://github.com/jackc602/Fallacy-Detector.git
cd Fallacy-Detector
uv sync
```

## Running

Each classifier is a standalone script that loads the data splits, trains,
evaluates on dev + test, and (for some models) saves a checkpoint:

```bash
uv run python src/modeling/tfidf_classifier.py
uv run python src/modeling/dan_classifier.py
uv run python src/modeling/dan_plus_tfidf_classifier.py
uv run python src/modeling/train_argument_features.py
uv run python src/modeling/train_tfidf_argument.py
uv run python src/modeling/train_bert_baseline.py
```


## Dependencies

- `torch`
- `numpy`
- `pandas`
- `pyarrow`
- `scikit-learn`
- `matplotlib`
- `transformers`
- `sentence-transformers`
- `datasets`
- `python-dotenv`
