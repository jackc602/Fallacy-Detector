import json
import argparse
import time
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
    samples = []
    seen = set()
    for _, row in df.iterrows():
        lab = row['logical_fallacies']
        if lab not in seen:
            samples.append({
                'text': row['source_article'],
                'label': lab,
                'fol': '',
            })
            seen.add(lab)
        if len(samples) >= 13:
            break
    return samples



def load_samples_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)



def translate(args):
    client = LLMClient(
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
    )
    args.model = client.model 

    # Load samples
    input_path = Path(args.input)
    if input_path.suffix == '.parquet':
        samples = load_samples_from_parquet(input_path)
    else:
        samples = load_samples_from_json(input_path)

    out_path = Path(args.output)
    tag = f"{args.backend}_{args.model.replace(':', '-').replace('/', '-')}"
    out_file = out_path.parent / f"{out_path.stem}_{tag}{out_path.suffix}"

    print(f"{args.backend} - {args.model}")
    print(f"Input: {input_path} - {len(samples)} samples")
    print(f"{out_file}")

    for i, sample in enumerate(samples):
        if sample['fol'].strip() and not args.overwrite:
            print(f"[{i+1}/{len(samples)}] Skipping (already has FOL): {sample['label']}")
            continue

        prompt = build_prompt(sample['text'], sample['label'])
        try:
            fol = client.chat(prompt, system_prompt=SYSTEM_PROMPT).strip()
            sample['fol'] = fol
            print(f"[{i+1}/{len(samples)}] {sample['label']}")
            print(f"  → {fol[:120]}{'...' if len(fol) > 120 else ''}")
        except Exception as e:
            print(f"[{i+1}/{len(samples)}] ERROR on {sample['label']}: {e}")
            sample['fol'] = f"ERROR: {e}"

        # Save after each sample so progress isn't lost
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        if args.backend != 'ollama':
            time.sleep(0.5)

    print(f"\nDone. Results saved to {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate fallacy samples to FOL')
    parser.add_argument('--backend', default='ollama', choices=['ollama', 'openai', 'gemini'])
    parser.add_argument('--model', default=None, help='Model name (defaults per backend)')
    parser.add_argument('--api-key', default=None, help='API key (or use env var)')
    parser.add_argument('--input', default='fol_test_samples.json', help='Input file (.json or .parquet)')
    parser.add_argument('--output', default='fol_results.json', help='Output file path')
    parser.add_argument('--overwrite', action='store_true', help='Re-translate even if FOL exists')
    args = parser.parse_args()
    translate(args)