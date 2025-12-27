# filepath: scripts/catalan_projection/run_catalan_projection.py
import os
import subprocess

def main():
    """
    Main orchestrator script to run the full projection pipeline.
    """
    print("--- Starting Catalan Metaphor Projection Pipeline ---")

    # --- Configuration ---
    # Path to the fine-tuned model. You must train this first using the repo's scripts.
    # e.g., './models/detection/es_model'
    MODEL_PATH = "bert-base-multilingual-cased" # Replace with your actual trained model path
    
    # Source language data (Spanish)
    SOURCE_PREMISE_FILE = "data/meta4xnli/detection/source_datasets/es/esxnli_prem.tsv"
    SOURCE_HYPOTHESIS_FILE = "data/meta4xnli/detection/source_datasets/es/esxnli_hyp.tsv"

    # Target language data (Catalan) - YOU NEED TO CREATE THESE FILES
    # Assumes you have downloaded xnli-ca and formatted it like the other datasets.
    CATALAN_PREMISE_FILE = "data/meta4xnli/detection/source_datasets/ca/xnli_ca_prem.tsv"
    CATALAN_HYPOTHESIS_FILE = "data/meta4xnli/detection/source_datasets/ca/xnli_ca_hyp.tsv"

    # Intermediate and final output paths
    PREDICTED_PREMISE_LABELS = "output/predictions_es/prem_labels.tsv"
    PREDICTED_HYPOTHESIS_LABELS = "output/predictions_es/hyp_labels.tsv"
    PROJECTED_PREMISE_FILE = "output/projected_ca/meta4xnli_ca_prem.tsv"
    PROJECTED_HYPOTHESIS_FILE = "output/projected_ca/meta4xnli_ca_hyp.tsv"

    # --- Step 1: Run Prediction on Source Language (Spanish) ---
    print("\n--- Step 1: Predicting metaphors on Spanish dataset ---")
    
    # Create directories for Catalan data if they don't exist
    os.makedirs("data/meta4xnli/detection/source_datasets/ca", exist_ok=True)
    # You would need to populate the catalan files here.
    if not os.path.exists(CATALAN_PREMISE_FILE) or not os.path.exists(CATALAN_HYPOTHESIS_FILE):
        print(f"Error: Catalan data files not found.")
        print(f"Please create '{CATALAN_PREMISE_FILE}' and '{CATALAN_HYPOTHESIS_FILE}'.")
        print("These should contain the raw Catalan sentences from the xnli-ca dataset, one per line.")
        return

    # Predict on premises
    subprocess.run([
        "python", "scripts/catalan_projection/1_run_prediction.py",
        "--model_name_or_path", MODEL_PATH,
        "--input_file", SOURCE_PREMISE_FILE,
        "--output_file", PREDICTED_PREMISE_LABELS
    ], check=True)

    # Predict on hypotheses
    subprocess.run([
        "python", "scripts/catalan_projection/1_run_prediction.py",
        "--model_name_or_path", MODEL_PATH,
        "--input_file", SOURCE_HYPOTHESIS_FILE,
        "--output_file", PREDICTED_HYPOTHESIS_LABELS
    ], check=True)

    print("\n--- Step 2: Projecting labels to Catalan ---")

    # Project labels for premises
    subprocess.run([
        "python", "scripts/catalan_projection/2_project_labels.py",
        "--source_sents_file", SOURCE_PREMISE_FILE,
        "--target_sents_file", CATALAN_PREMISE_FILE,
        "--source_labels_file", PREDICTED_PREMISE_LABELS,
        "--output_file", PROJECTED_PREMISE_FILE
    ], check=True)

    # Project labels for hypotheses
    subprocess.run([
        "python", "scripts/catalan_projection/2_project_labels.py",
        "--source_sents_file", SOURCE_HYPOTHESIS_FILE,
        "--target_sents_file", CATALAN_HYPOTHESIS_FILE,
        "--source_labels_file", PREDICTED_HYPOTHESIS_LABELS,
        "--output_file", PROJECTED_HYPOTHESIS_FILE
    ], check=True)

    print("\n--- Pipeline Finished Successfully! ---")
    print(f"Projected Catalan premise data at: {PROJECTED_PREMISE_FILE}")
    print(f"Projected Catalan hypothesis data at: {PROJECTED_HYPOTHESIS_FILE}")
    print("\nNext step: Manually validate the projected labels.")


if __name__ == "__main__":
    main()
