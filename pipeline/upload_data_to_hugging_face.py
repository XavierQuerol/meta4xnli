"""
Script to upload TSV metaphor detection dataset to Hugging Face
Run this script locally or in Colab to prepare and upload your dataset
"""

import os
import glob
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

# ==================== CONFIGURATION ====================
# UPDATE THESE PATHS AND NAMES
TSV_FILES_PATH = "/path/to/your/tsv/files/*.tsv"  # Path to your TSV files
DATASET_NAME = "your-username/metaphor-catalan"    # Your HF username/dataset-name
# =======================================================

def read_tsv_file(file_path):
    """
    Read a single TSV file and return tokens and labels.
    Sentences are separated by empty lines.
    """
    sentences = []
    labels = []
    current_tokens = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line == "":  # Empty line = end of sentence
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    current_tokens.append(token)
                    current_labels.append(label)
        
        # Add last sentence if exists
        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_labels)
    
    return sentences, labels

def load_all_tsv_files(pattern):
    """
    Load all TSV files matching the pattern.
    """
    all_sentences = []
    all_labels = []
    
    files = glob.glob(pattern)
    print(f"Found {len(files)} TSV files")
    
    for file_path in sorted(files):
        print(f"Loading {os.path.basename(file_path)}...")
        sentences, labels = read_tsv_file(file_path)
        all_sentences.extend(sentences)
        all_labels.extend(labels)
        print(f"  → Loaded {len(sentences)} sentences")
    
    return all_sentences, all_labels

def create_dataset():
    """
    Load TSV files and create a Hugging Face dataset.
    """
    print("\n" + "="*60)
    print("Loading TSV files...")
    print("="*60)
    
    sentences, labels = load_all_tsv_files(TSV_FILES_PATH)
    print(f"\n✓ Total sentences loaded: {len(sentences)}")
    
    # Create label mapping
    unique_labels = set()
    for label_seq in labels:
        unique_labels.update(label_seq)
    
    label_list = sorted(list(unique_labels))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    print(f"✓ Labels found: {label_list}")
    
    # Convert labels to IDs
    label_ids = [[label2id[label] for label in label_seq] for label_seq in labels]
    
    # Create dataset
    data_dict = {
        "tokens": sentences,
        "ner_tags": label_ids
    }
    
    dataset = Dataset.from_dict(data_dict)
    
    # Add features with label names
    from datasets import ClassLabel, Sequence
    features = dataset.features.copy()
    features["ner_tags"] = Sequence(ClassLabel(names=label_list))
    dataset = dataset.cast(features)
    
    print(f"✓ Dataset created with {len(dataset)} examples")
    
    # Split into train, validation, and test
    print("\n" + "="*60)
    print("Splitting dataset...")
    print("="*60)
    
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=42)
    
    dataset_dict = DatasetDict({
        'train': train_test['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })
    
    print(f"✓ Train: {len(dataset_dict['train'])} examples")
    print(f"✓ Validation: {len(dataset_dict['validation'])} examples")
    print(f"✓ Test: {len(dataset_dict['test'])} examples")
    
    return dataset_dict, label_list

def upload_to_hub(dataset_dict, label_list):
    """
    Upload dataset to Hugging Face Hub.
    """
    print("\n" + "="*60)
    print("Uploading to Hugging Face Hub...")
    print("="*60)
    
    # You'll be prompted to log in if not already logged in
    from huggingface_hub import login
    login()
    
    # Push to hub
    dataset_dict.push_to_hub(DATASET_NAME)
    
    print(f"\n✓ Dataset uploaded successfully to: https://huggingface.co/datasets/{DATASET_NAME}")
    print("\nYou can now load it with:")
    print(f'  dataset = load_dataset("{DATASET_NAME}")')
    
    # Print example README content
    print("\n" + "="*60)
    print("Suggested README.md content for your dataset:")
    print("="*60)
    print(f"""
---
language:
- ca
task_categories:
- token-classification
tags:
- metaphor-detection
- catalan
size_categories:
- n<1K
---

# Catalan Metaphor Detection Dataset

## Dataset Description

This dataset contains Catalan text annotated for metaphor detection at the token level.

### Labels

{', '.join(label_list)}

### Data Splits

- Train: {len(dataset_dict['train'])} examples
- Validation: {len(dataset_dict['validation'])} examples  
- Test: {len(dataset_dict['test'])} examples

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{DATASET_NAME}")
```

## Dataset Structure

Each example contains:
- `tokens`: List of tokens in the sentence
- `ner_tags`: List of labels (0 = O, 1 = B-METAPHOR)
""")

def main():
    """
    Main function to create and upload dataset.
    """
    print("\n" + "="*60)
    print("TSV to Hugging Face Dataset Uploader")
    print("="*60)
    
    # Create dataset
    dataset_dict, label_list = create_dataset()
    
    # Show example
    print("\n" + "="*60)
    print("Example from dataset:")
    print("="*60)
    example = dataset_dict['train'][0]
    print(f"Tokens: {example['tokens']}")
    print(f"Labels: {example['ner_tags']}")
    
    # Ask for confirmation
    print("\n" + "="*60)
    response = input(f"Upload to {DATASET_NAME}? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        upload_to_hub(dataset_dict, label_list)
    else:
        print("Upload cancelled.")
        
        # Option to save locally
        save_local = input("Save locally instead? (yes/no): ")
        if save_local.lower() in ['yes', 'y']:
            local_path = "./metaphor_dataset"
            dataset_dict.save_to_disk(local_path)
            print(f"✓ Dataset saved to {local_path}")

if __name__ == "__main__":
    main()