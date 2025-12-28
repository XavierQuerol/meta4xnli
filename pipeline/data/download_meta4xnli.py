from datasets import load_dataset, concatenate_datasets
import os

data_files = [
    "det_es_finetune",
    "det_en_finetune",
    "det_es_eval",
    "det_en_eval"
]

output_dir = "pipeline/data/meta4xnli"
os.makedirs(output_dir, exist_ok=True)

for data in data_files:
    # Load dataset config
    dataset_dict = load_dataset("HiTZ/meta4xnli", data)

    # Concatenate all splits into one dataset
    all_splits = concatenate_datasets(list(dataset_dict.values()))
    all_splits = all_splits.select_columns(["id", "tokens", "tags"])

    output_path = os.path.join(output_dir, f"{data}.csv")

    # Save to CSV
    all_splits.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
