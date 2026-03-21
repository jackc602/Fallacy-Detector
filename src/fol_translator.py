import json
from ollama_integration import OllamaClient

def translate_to_fol():
    # Load the samples
    with open('fol_test_samples.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Initialize Ollama client (change model as needed)
    client = OllamaClient(model='mistral:latest')

    # Prompt template for FOL translation
    prompt_template = """
Translate this logical fallacy example to first-order logic (FOL). Provide only the FOL formula without additional explanation.

Fallacy type: {label}
Text: {text}

FOL translation:
"""

    updated = False
    for i, sample in enumerate(samples):
        if not sample['fol'].strip():  # Only process if fol is empty
            prompt = prompt_template.format(label=sample['label'], text=sample['text'])
            try:
                fol_translation = client.chat(prompt).strip()
                sample['fol'] = fol_translation
                updated = True
                print(f"Translated sample {i+1}: {fol_translation}")
            except Exception as e:
                print(f"Error translating sample {i+1}: {e}")
                sample['fol'] = f"Error: {e}"

    if updated:
        # Save updated samples
        with open('fol_test_samples.json', 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print("Updated fol_test_samples.json with translations")
    else:
        print("No updates needed - all samples already have FOL translations")

if __name__ == "__main__":
    translate_to_fol()