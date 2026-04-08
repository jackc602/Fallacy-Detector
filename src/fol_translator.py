import re
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from llm_client import LLMClient


SYSTEM_PROMPT = """\
You are a formal logic expert. Your task is to translate natural language arguments \
into structured first-order logic (FOL).

For each argument, identify and separately label:
  1. All explicitly stated premises  →  "Premise N: <formula>"
  2. Any unstated assumption needed  →  "Hidden assumption: <formula>"
  3. The conclusion being drawn      →  "Conclusion: <formula>"

Rules:
- Use standard FOL notation: ∀, ∃, →, ∧, ∨, ¬, ↔
- Use descriptive CamelCase predicate names: Person(x), GoesSwimming(joe)
- Use lowercase constants for named individuals: joe, kentucky, randall
- Every response MUST end with a "Conclusion:" line
- If the argument is a simple claim with no explicit structure, use "Proposition: <formula>"
- Output ONLY the labeled formulas — no prose, no definitions, no commentary
"""

# Few-shot examples: label shown so model learns format/style, not given for the query item
FEW_SHOT = [
    {
        "text": "I know five people from Kentucky. They are all racists. Therefore, Kentuckians are racist.",
        "fol": (
            "Premise 1: Kentuckian(a) ∧ Kentuckian(b) ∧ Kentuckian(c) ∧ Kentuckian(d) ∧ Kentuckian(e)\n"
            "Premise 2: Racist(a) ∧ Racist(b) ∧ Racist(c) ∧ Racist(d) ∧ Racist(e)\n"
            "Hidden assumption: ∃x (Kentuckian(x) ∧ Racist(x)) → ∀x (Kentuckian(x) → Racist(x))\n"
            "Conclusion: ∀x (Kentuckian(x) → Racist(x))"
        ),
    },
    {
        "text": "We either have to cut taxes or leave a huge debt for our children.",
        "fol": (
            "Premise: CutTaxes ∨ LeaveDebt\n"
            "Hidden assumption: ∀x (FeasibleOption(x) → (x = CutTaxes ∨ x = LeaveDebt))\n"
            "Conclusion: ¬CutTaxes → LeaveDebt"
        ),
    },
    {
        "text": "Since many people believe this, then it must be true.",
        "fol": (
            "Premise: BelievedByMany(p)\n"
            "Hidden assumption: ∀x (BelievedByMany(x) → True(x))\n"
            "Conclusion: True(p)"
        ),
    },
]

# Markers indicating valid FOL content
_FOL_SYMBOLS = frozenset(['∀', '∃', '→', '∧', '∨', '¬', '↔'])
_SECTION_LABELS = ('Premise', 'Conclusion', 'Proposition', 'Hidden assumption')
_PROSE_STARTS = (
    "i ", "i'll", "here ", "here's", "the argument", "let me",
    "this argument", "the text", "we can", "note:", "sure",
    "of course", "certainly", "to translate",
)

# Maximum retries when the model returns malformed output
MAX_FORMAT_RETRIES = 2


def validate_fol_response(text: str) -> tuple[bool, str]:
    """Return (is_valid, reason). Checks that the response looks like real FOL."""
    stripped = text.strip()

    if not stripped:
        return False, "empty response"

    if len(stripped) < 10:
        return False, "response too short"

    first_line = stripped.split('\n')[0].lower().strip()
    if any(first_line.startswith(p) for p in _PROSE_STARTS):
        return False, f"starts with prose: {first_line[:40]!r}"

    has_symbol = any(s in stripped for s in _FOL_SYMBOLS)
    has_label = any(stripped.startswith(lbl) or f'\n{lbl}' in stripped for lbl in _SECTION_LABELS)
    has_predicate = bool(re.search(r'[A-Z][a-zA-Z]+\(', stripped))

    if not (has_symbol or has_label or has_predicate):
        return False, "no FOL content detected (no symbols, labels, or predicates)"

    has_premise = 'Premise' in stripped or 'Hidden assumption' in stripped
    has_conclusion = 'Conclusion:' in stripped or 'Proposition:' in stripped
    if has_premise and not has_conclusion:
        return False, "truncated: has premises but no Conclusion line"

    return True, "ok"


def build_prompt(text: str) -> str:
    """Build the few-shot prompt. The label is intentionally NOT included in the query."""
    examples = ""
    for ex in FEW_SHOT:
        examples += f'Text: "{ex["text"]}"\nFOL:\n{ex["fol"]}\n\n'

    return (
        f"Here are some examples of FOL translations:\n\n"
        f"{examples}"
        f"Now translate this:\n\n"
        f'Text: "{text}"\n'
        f"FOL:\n"
    )


def build_strict_prompt(text: str) -> str:
    """Stricter fallback prompt used when the first attempt returned malformed output."""
    return (
        f"Translate the following argument into first-order logic.\n"
        f"Your response MUST begin with 'Premise 1:' or 'Proposition:'.\n"
        f"Use FOL symbols: ∀, ∃, →, ∧, ∨, ¬. No prose or explanation.\n\n"
        f'Text: "{text}"\n'
        f"FOL:\n"
    )


def load_samples_from_parquet(parquet_path):
    """Load unique samples from a parquet file."""
    df = pd.read_parquet(parquet_path)
    return [
        {'text': row['source_article'], 'label': row['logical_fallacies'], 'fol': ''}
        for row in df.to_dict('records')
    ]


def load_samples_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_existing_fol(out_file):
    """Returns dict of text -> fol for resuming a previous run."""
    if not out_file.exists():
        return {}
    with open(out_file, 'r', encoding='utf-8') as f:
        existing = json.load(f)
    return {s['text']: s['fol'] for s in existing if s.get('fol', '').strip()}


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

    # pre-populate fol from existing output file
    existing = load_existing_fol(out_file)
    for s in samples:
        if s['text'] in existing:
            s['fol'] = existing[s['text']]

    todo = [s for s in samples if not s.get('fol', '').strip() or args.overwrite]

    print(f"{args.backend} - {args.model}")
    print(f"Input:  {input_path} — {len(samples)} samples")
    print(f"Output: {out_file}")
    print(f"  {len(samples) - len(todo)} already done, {len(todo)} to process")

    if not todo:
        print("Nothing to do.")
        return

    lock = threading.Lock()
    completed_count = [0]
    error_counts = {'api': 0, 'format': 0}

    def process(sample):
        text = sample['text']
        fol = None
        last_error = None

        for attempt in range(MAX_FORMAT_RETRIES):
            prompt = build_prompt(text) if attempt == 0 else build_strict_prompt(text)
            try:
                raw = client.chat(prompt, system_prompt=SYSTEM_PROMPT).strip()
            except Exception as e:
                last_error = f"API:{type(e).__name__}: {e}"
                with lock:
                    error_counts['api'] += 1
                break

            is_valid, reason = validate_fol_response(raw)
            if is_valid:
                fol = raw
                break
            else:
                last_error = f"FORMAT:{reason}"
                if attempt < MAX_FORMAT_RETRIES - 1:
                    print(f"  [format-retry] {text[:60]!r} — {reason}")

        if fol is None:
            fol = f"ERROR:{last_error}"
            if last_error and last_error.startswith('FORMAT'):
                with lock:
                    error_counts['format'] += 1

        sample['fol'] = fol

        with lock:
            completed_count[0] += 1
            n = completed_count[0]
            if n % args.save_every == 0:
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
                print(f"  Checkpoint saved ({n}/{len(todo)} done, "
                      f"api_err={error_counts['api']}, fmt_err={error_counts['format']})")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, s) for s in todo]
        for fut in as_completed(futures):
            fut.result()  
            
    # Final save
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    valid = sum(1 for s in samples if s.get('fol') and not s['fol'].startswith('ERROR'))
    print(f"\nDone. {valid}/{len(samples)} valid FOL translations.")
    print(f"API errors: {error_counts['api']}  |  Format errors: {error_counts['format']}")
    print(f"Results saved to {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate fallacy samples to FOL')
    parser.add_argument('--backend', default='gemini', choices=['ollama', 'openai', 'gemini'])
    parser.add_argument('--model', default='gemini-2.5-pro', help='Model name (defaults per backend)')
    parser.add_argument('--api-key', default=None, help='API key (or use env var)')
    parser.add_argument('--input', required=True, help='Input file (.json or .parquet)')
    parser.add_argument('--output', default='fol_results.json', help='Output file path')
    parser.add_argument('--overwrite', action='store_true', help='Re-translate even if FOL exists')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel API calls')
    parser.add_argument('--save-every', type=int, default=50, dest='save_every',
                        help='Checkpoint frequency in samples')
    args = parser.parse_args()
    translate(args)
