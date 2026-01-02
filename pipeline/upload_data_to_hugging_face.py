"""
Script to upload TSV metaphor detection dataset to Hugging Face.

Run this script locally or in Colab to prepare and upload your dataset.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

from datasets import Dataset, DatasetDict

# ==================== CONFIGURATION ====================
# UPDATE THESE PATHS AND NAMES
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSV_INPUT = str(
    REPO_ROOT
    / "data"
    / "meta4xnli"
    / "detection"
    / "final_projected_labels"
    / "xnli_dev_hyp.tsv"
)
TSV_FILES_PATH = DEFAULT_TSV_INPUT  # File/dir/glob pattern
DATASET_NAME = "mariadelcarmenramirez/metaphor-catalan-iter1"  # Your HF username/dataset-name
# =======================================================


def read_tsv_file(file_path: str):
    """
    Read a single TSV file and return tokens and labels.
    Sentences are separated by empty lines.
    """
    sentences: list[list[str]] = []
    labels: list[list[str]] = []
    current_tokens: list[str] = []
    current_labels: list[str] = []

    with open(file_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if line == "":  # Empty line = end of sentence
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue
            token, label = parts
            current_tokens.append(token)
            current_labels.append(label)

    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)

    return sentences, labels


def _expand_tsv_inputs(tsv_input: str) -> list[str]:
    """
    Expand a file/dir/glob into a list of TSV files.
    """
    if not tsv_input:
        return []

    expanded = os.path.expandvars(os.path.expanduser(tsv_input))

    if any(ch in expanded for ch in ["*", "?", "["]):
        return [p for p in sorted(glob.glob(expanded)) if os.path.isfile(p)]

    path = Path(expanded)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return [str(p) for p in sorted(path.glob("*.tsv")) if p.is_file()]

    return []


def load_all_tsv_files(tsv_input: str):
    """
    Load all TSV files from a file/dir/glob input.
    """
    all_sentences: list[list[str]] = []
    all_labels: list[list[str]] = []

    files = _expand_tsv_inputs(tsv_input)
    print(f"Found {len(files)} TSV files")

    for file_path in files:
        print(f"Loading {os.path.basename(file_path)}...")
        sentences, labels = read_tsv_file(file_path)
        all_sentences.extend(sentences)
        all_labels.extend(labels)
        print(f"  Loaded {len(sentences)} sentences")

    return all_sentences, all_labels


def create_dataset(tsv_input: str):
    """
    Load TSV files and create a Hugging Face dataset.
    """
    print("\n" + "=" * 60)
    print("Loading TSV files...")
    print("=" * 60)

    sentences, labels = load_all_tsv_files(tsv_input)
    print(f"\nTotal sentences loaded: {len(sentences)}")
    if not sentences:
        raise FileNotFoundError(
            "No sentences loaded. Check your `--tsv` path (file/dir/glob). "
            f"Got: {tsv_input!r}"
        )

    unique_labels: set[str] = set()
    for label_seq in labels:
        unique_labels.update(label_seq)

    # Ensure stable, expected ids: O -> 0, B-METAPHOR -> 1 (when present).
    preferred_order = ["O", "B-METAPHOR"]
    label_list: list[str] = [l for l in preferred_order if l in unique_labels]
    label_list.extend(sorted(l for l in unique_labels if l not in set(preferred_order)))
    label2id = {label: i for i, label in enumerate(label_list)}
    print(f"Labels found: {label_list}")

    label_ids = [[label2id[label] for label in label_seq] for label_seq in labels]

    example_ids = list(range(len(sentences)))
    data_dict = {"id": example_ids, "tokens": sentences, "tags": label_ids}
    dataset = Dataset.from_dict(data_dict)

    from datasets import ClassLabel, Sequence

    features = dataset.features.copy()
    features["tags"] = Sequence(ClassLabel(names=label_list))
    dataset = dataset.cast(features)

    print(f"Dataset created with {len(dataset)} examples")

    print("\n" + "=" * 60)
    print("Splitting dataset...")
    print("=" * 60)

    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=42)

    dataset_dict = DatasetDict(
        {
            "train": train_test["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"],
        }
    )

    print(f"Train: {len(dataset_dict['train'])} examples")
    print(f"Validation: {len(dataset_dict['validation'])} examples")
    print(f"Test: {len(dataset_dict['test'])} examples")

    return dataset_dict, label_list


def upload_to_hub(dataset_dict: DatasetDict, dataset_name: str, label_list: list[str]):
    """
    Upload dataset to Hugging Face Hub.
    """
    print("\n" + "=" * 60)
    print("Uploading to Hugging Face Hub...")
    print("=" * 60)

    from huggingface_hub import login

    login()
    dataset_dict.push_to_hub(dataset_name)

    print(f"\nDataset uploaded successfully to: https://huggingface.co/datasets/{dataset_name}")
    print("\nYou can now load it with:")
    print(f'  dataset = load_dataset("{dataset_name}")')

    print("\n" + "=" * 60)
    print("Suggested README.md content for your dataset:")
    print("=" * 60)
    print(
        f"""
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

dataset = load_dataset("{dataset_name}")
```

## Dataset Structure

Each example contains:
- `tokens`: List of tokens in the sentence
- `tags`: List of labels (0 = O, 1 = B-METAPHOR)
"""
    )


def main():
    print("\n" + "=" * 60)
    print("TSV to Hugging Face Dataset Uploader")
    print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv",
        default=TSV_FILES_PATH,
        help="Path to TSV file, a directory of .tsv files, or a glob pattern.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help='Hugging Face dataset repo id, e.g. "username/dataset_name".',
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to the Hub without prompting (requires auth).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var). If omitted, an interactive login prompt is used.",
    )
    args = parser.parse_args()

    dataset_dict, label_list = create_dataset(args.tsv)

    print("\n" + "=" * 60)
    print("Example from dataset:")
    print("=" * 60)
    example = dataset_dict["train"][0]
    print(f"Id: {example['id']}")
    print(f"Tokens: {example['tokens']}")
    print(f"Labels: {example['tags']}")

    if args.upload:
        from huggingface_hub import login

        login(token=args.token or os.environ.get("HF_TOKEN"))
        dataset_dict.push_to_hub(args.dataset_name)
        print(f"\nDataset uploaded successfully to: https://huggingface.co/datasets/{args.dataset_name}")
        return

    print("\n" + "=" * 60)
    response = input(f"Upload to {args.dataset_name}? (yes/no): ")
    if response.lower() in ["yes", "y"]:
        if args.token or os.environ.get("HF_TOKEN"):
            from huggingface_hub import login

            login(token=args.token or os.environ.get("HF_TOKEN"))
            dataset_dict.push_to_hub(args.dataset_name)
            print(f"\nDataset uploaded successfully to: https://huggingface.co/datasets/{args.dataset_name}")
        else:
            upload_to_hub(dataset_dict, args.dataset_name, label_list)
        return

    print("Upload cancelled.")
    save_local = input("Save locally instead? (yes/no): ")
    if save_local.lower() in ["yes", "y"]:
        local_path = "./metaphor_dataset"
        dataset_dict.save_to_disk(local_path)
        print(f"Dataset saved to {local_path}")


if __name__ == "__main__":
    main()
