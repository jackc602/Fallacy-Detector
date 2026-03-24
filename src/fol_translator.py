import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from llm_client import LLMClient


SYSTEM_PROMPT = """\
You are a formal logic expert. Your task is to translate natural language arguments \
into well-formed first-order logic (FOL) formulas.

Rules:
- Use standard FOL notation: ∀, ∃, →, ∧, ∨, ¬, ↔
- Use descriptive predicate names: e.g., Student(x), Prepares(x, exam)
- Use constants for named entities: e.g., kentucky, brawndo
- Output ONLY the FOL formula(s). No explanations, no variable definitions, no commentary.
- If the argument has multiple premises and a conclusion, write them as:
  Premise 1: <formula>
  Premise 2: <formula>
  Conclusion: <formula>
- Keep it concise. One fallacious argument = a few lines of FOL at most.
"""

# Format and style for FOL 
FEW_SHOT = [
    {
        "text": "I know five people from Kentucky. They are all racists. Therefore, Kentuckians are racist.",
        "label": "faulty generalization",
        "fol": (
            "Premise 1: Person(a) ∧ Person(b) ∧ Person(c) ∧ Person(d) ∧ Person(e)\n"
            "Premise 2: From(a, kentucky) ∧ From(b, kentucky) ∧ From(c, kentucky) ∧ From(d, kentucky) ∧ From(e, kentucky)\n"
            "Premise 3: Racist(a) ∧ Racist(b) ∧ Racist(c) ∧ Racist(d) ∧ Racist(e)\n"
            "Conclusion: ∀x (From(x, kentucky) → Racist(x))"
        ),
    },
    {
        "text": "We either have to cut taxes or leave a huge debt for our children.",
        "label": "false dilemma",
        "fol": (
            "Premise: CutTaxes ∨ LeaveDebt\n"
            "Hidden assumption: ¬∃x (Option(x) ∧ x ≠ CutTaxes ∧ x ≠ LeaveDebt)"
        ),
    },
    {
        "text": "Argues that because something is popular, it must be right.",
        "label": "ad populum",
        "fol": "Premise: Popular(p) → Right(p)",
    },
]


def build_prompt(text, label):
    examples = ""
    for ex in FEW_SHOT:
        examples += f'Text: "{ex["text"]}"\nFallacy type: {ex["label"]}\nFOL:\n{ex["fol"]}\n\n'

    return (
        f"Here are some examples of FOL translations:\n\n"
        f"{examples}"
        f"Now translate this:\n\n"
        f'Text: "{text}"\n'
        f"Fallacy type: {label}\n"
        f"FOL:\n"
    )


def load_samples_from_parquet(parquet_path):
    """Getting unique samples from the parquet file, since it contains many duplicates."""
    df = pd.read_parquet(parquet_path)
    return [
        {'text': row['source_article'], 'label': row['logical_fallacies'], 'fol': ''}
        for row in df.to_dict('records')
    ]



def load_samples_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_existing_fol(out_file):
    """Returns dict of (text, label) -> fol for resuming a previous run."""
    if not out_file.exists():
        return {}
    with open(out_file, 'r', encoding='utf-8') as f:
        existing = json.load(f)
    return {(s['text'], s['label']): s['fol'] for s in existing if s.get('fol', '').strip()}


def translate(args):
    client = LLMClient(
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
    )

    # Load samples
    input_path = Path(args.input)
    if input_path.suffix == '.parquet':
        samples = load_samples_from_parquet(input_path)
    else:
        samples = load_samples_from_json(input_path)

    out_path = Path(args.output)
    tag = f"{args.backend}_{str(args.model).replace(':', '-').replace('/', '-')}"
    out_file = out_path.parent / f"{out_path.stem}_{tag}{out_path.suffix}"

    # Resume: pre-populate fol from existing output file
    existing = load_existing_fol(out_file)
    for s in samples:
        if (s['text'], s['label']) in existing:
            s['fol'] = existing[(s['text'], s['label'])]

    todo = [s for s in samples if not s['fol'].strip() or args.overwrite]

    print(f"{args.backend} - {args.model}")
    print(f"Input: {input_path} - {len(samples)} samples")
    print(f"Output: {out_file}")
    print(f"  {len(samples) - len(todo)} already done, {len(todo)} to process")

    if not todo:
        print("Nothing to do.")
        return

    lock = threading.Lock()
    completed_count = [0]

    def process(sample):
        prompt = build_prompt(sample['text'], sample['label'])
        try:
            fol = client.chat(prompt, system_prompt=SYSTEM_PROMPT).strip()
        except Exception as e:
            fol = f"ERROR: {e}"
        sample['fol'] = fol
        with lock:
            completed_count[0] += 1
            n = completed_count[0]
            if n % args.save_every == 0:
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
                print(f"  Checkpoint saved ({n} done)")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, s) for s in todo]
        for fut in as_completed(futures):
            fut.result()  # re-raise any exceptions

    # Final save
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Results saved to {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate fallacy samples to FOL')
    parser.add_argument('--backend', default='ollama', choices=['ollama', 'openai', 'gemini'])
    parser.add_argument('--model', default=None, help='Model name (defaults per backend)')
    parser.add_argument('--api-key', default=None, help='API key (or use env var)')
    parser.add_argument('--input', default='fol_test_samples.json', help='Input file (.json or .parquet)')
    parser.add_argument('--output', default='fol_results.json', help='Output file path')
    parser.add_argument('--overwrite', action='store_true', help='Re-translate even if FOL exists')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel API calls')
    parser.add_argument('--save-every', type=int, default=20, dest='save_every', help='Checkpoint frequency in samples')
    args = parser.parse_args()
    translate(args)