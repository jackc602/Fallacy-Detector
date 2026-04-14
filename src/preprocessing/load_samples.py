import json
from datasets import load_dataset

# Load the dev split of the dataset
dataset = load_dataset('tasksource/logical-fallacy', split='dev')

# Collect samples with variety in fallacy types
samples = []
seen_labels = set()

for example in dataset:
    label = example['logical_fallacies']
    if label not in seen_labels:
        samples.append({
            'text': example['source_article'],
            'label': label,
            'fol': ''
        })
        seen_labels.add(label)
    if len(samples) >= 13:  # Only 13 unique types available
        break

# Save to JSON file
with open('fol_test_samples.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

# Print samples to console
for i, sample in enumerate(samples, 1):
    print(f"Sample {i}:")
    print(f"Label: {sample['label']}")
    print(f"Text: {sample['text']}")
    print()