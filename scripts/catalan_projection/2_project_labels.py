# filepath: scripts/catalan_projection/2_project_labels.py
import argparse
import os
import torch
import awesome_align as align

def project_labels(source_sents_file, target_sents_file, source_labels_file, output_file):
    """
    Projects BIO labels from a source language file to a target language file.
    """
    print("Loading alignment model...")
    align.AwesomeAlign.from_pretrained('bert-base-multilingual-cased')
    
    print("Reading source sentences, target sentences, and source labels...")
    with open(source_sents_file, 'r', encoding='utf-8') as f:
        source_sents = [line.strip() for line in f]
    with open(target_sents_file, 'r', encoding='utf-8') as f:
        target_sents = [line.strip() for line in f]
    with open(source_labels_file, 'r', encoding='utf-8') as f:
        # Assuming the prediction script saves "sentence \t B-TAG I-TAG O..."
        source_labels_data = [line.strip().split('\t')[1].split() for line in f]

    assert len(source_sents) == len(target_sents) == len(source_labels_data), \
        "Mismatch in number of lines between input files."

    print(f"Aligning and projecting labels for {len(source_sents)} sentence pairs...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(len(source_sents)):
            src_sent = source_sents[i]
            tgt_sent = target_sents[i]
            src_labels = source_labels_data[i]
            
            # Get alignments
            try:
                alignment_str = align.align(src_sent, tgt_sent)
                alignments = [tuple(map(int, x.split('-'))) for x in alignment_str.split()]
            except Exception as e:
                print(f"Could not align sentence pair {i}: {e}")
                print(f"SRC: {src_sent}")
                print(f"TGT: {tgt_sent}")
                continue

            src_tokens = src_sent.split()
            tgt_tokens = tgt_sent.split()
            
            if len(src_tokens) != len(src_labels):
                print(f"Warning: Token count mismatch in source sentence {i}. Skipping.")
                continue

            # Initialize target labels as 'O'
            tgt_labels = ['O'] * len(tgt_tokens)

            # Create a map from source index to target indices
            src_to_tgt_map = {}
            for src_idx, tgt_idx in alignments:
                if src_idx not in src_to_tgt_map:
                    src_to_tgt_map[src_idx] = []
                src_to_tgt_map[src_idx].append(tgt_idx)

            # Project labels
            for src_idx, label in enumerate(src_labels):
                if label != 'O':
                    if src_idx in src_to_tgt_map:
                        for tgt_idx in src_to_tgt_map[src_idx]:
                            if tgt_idx < len(tgt_labels):
                                # Direct projection: B-MET -> B-MET, I-MET -> I-MET
                                tgt_labels[tgt_idx] = label
            
            # Write the Catalan sentence and its new projected labels
            f_out.write(f"{tgt_sent}\t{' '.join(tgt_labels)}\n")

    print(f"Projection finished. Output written to '{output_file}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Project BIO labels from source to target language.")
    parser.add_argument("--source_sents_file", type=str, required=True, help="Path to source language sentences.")
    parser.add_argument("--target_sents_file", type=str, required=True, help="Path to parallel target language sentences (Catalan).")
    parser.add_argument("--source_labels_file", type=str, required=True, help="Path to file with predicted BIO labels for source sentences.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the projected labels for the target language.")

    args = parser.parse_args()
    project_labels(args.source_sents_file, args.target_sents_file, args.source_labels_file, args.output_file)
