import streamlit as st
import pandas as pd
import os
from collections import defaultdict

# Page configuration
st.set_page_config(layout="wide", page_title="Meta-eval")

# --- PATHS ---
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_PATH, "data", "meta4xnli", "detection")

# Source data
EN_PATH = os.path.join(DATA_PATH, "source_datasets", "en")
ES_PATH = os.path.join(DATA_PATH, "source_datasets", "es")
CA_PATH = os.path.join(DATA_PATH, "source_datasets", "ca")

# Projected data
CA_EN_PROJ_PATH = os.path.join(DATA_PATH, "projected_labels", "ca-en")
CA_ES_PROJ_PATH = os.path.join(DATA_PATH, "projected_labels", "ca-es")

# Output data
FINAL_LABELS_PATH = os.path.join(DATA_PATH, "final_projected_labels")
os.makedirs(FINAL_LABELS_PATH, exist_ok=True)

# --- DATA LOADING ---

def load_sentences_from_file(file_path):
    """Reads a file with one token and label per line, separated by newlines."""
    sentences = []
    if not os.path.exists(file_path):
        return sentences
    with open(file_path, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append({"words": words, "labels": labels})
                    words = []
                    labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    word, label = parts
                    words.append(word)
                    labels.append(label)
        if words:
            sentences.append({"words": words, "labels": labels})
    return sentences

@st.cache_data
def load_all_data(dataset_name):
    """Loads all data for a given dataset name."""
    file_name = f"{dataset_name}.tsv"
    data = {
        "es": load_sentences_from_file(os.path.join(ES_PATH, file_name)),
        "en": load_sentences_from_file(os.path.join(EN_PATH, file_name)),
        "ca_es_proj": load_sentences_from_file(os.path.join(CA_ES_PROJ_PATH, file_name)),
        "ca_en_proj": load_sentences_from_file(os.path.join(CA_EN_PROJ_PATH, file_name)),
    }
    return data

def get_validated_ids(dataset_name):
    """Loads the set of already validated sentence IDs."""
    validated_file = os.path.join(FINAL_LABELS_PATH, f"{dataset_name}_validated.txt")
    validated_ids = set()
    if os.path.exists(validated_file):
        with open(validated_file, 'r', encoding='utf-8') as f:
            for line in f:
                validated_ids.add(int(line.strip()))
    return validated_ids

def get_doubtful_ids(dataset_name):
    """Loads the set of sentence IDs marked as doubtful."""
    doubtful_file = os.path.join(FINAL_LABELS_PATH, f"{dataset_name}_doubtful.txt")
    doubtful_ids = set()
    if os.path.exists(doubtful_file):
        with open(doubtful_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doubtful_ids.add(int(line.strip()))
                except ValueError:
                    pass # Ignore non-integer lines
    return doubtful_ids

def get_corrected_ids(dataset_name):
    """Loads the set of sentence IDs that have been retranslated."""
    corrected_file = os.path.join(FINAL_LABELS_PATH, f"{dataset_name}_corrected.txt")
    corrected_ids = set()
    if os.path.exists(corrected_file):
        with open(corrected_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    corrected_ids.add(int(line.strip()))
                except ValueError:
                    pass
    return corrected_ids

def save_validated_sentence(dataset_name, sentence_id, words, labels):
    """Saves a validated sentence and its ID."""
    # Save the labels
    output_file = os.path.join(FINAL_LABELS_PATH, f"{dataset_name}.tsv")
    with open(output_file, 'a', encoding='utf-8') as f:
        for word, label in zip(words, labels):
            f.write(f"{word}\t{label}\n")
        f.write("\n")

    # Save the ID
    validated_file = os.path.join(FINAL_LABELS_PATH, f"{dataset_name}_validated.txt")
    with open(validated_file, 'a', encoding='utf-8') as f:
        f.write(f"{sentence_id}\n")

def save_doubtful_id(dataset_name, sentence_id):
    """Saves a doubtful sentence ID to its file."""
    doubtful_file = os.path.join(FINAL_LABELS_PATH, f"{dataset_name}_doubtful.txt")
    with open(doubtful_file, 'a', encoding='utf-8') as f:
        f.write(f"{sentence_id}\n")

def save_retranslated_sentence(dataset_name, sentence_id, new_sentence):
    """Saves a retranslated sentence and its ID."""
    # Save the retranslated sentence
    output_file = os.path.join(FINAL_LABELS_PATH, f"{dataset_name}_retranslated.tsv")
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{sentence_id}\t{new_sentence}\n")

    # Save the ID to prevent it from being validated again
    corrected_file = os.path.join(FINAL_LABELS_PATH, f"{dataset_name}_corrected.txt")
    with open(corrected_file, 'a', encoding='utf-8') as f:
        f.write(f"{sentence_id}\n")


# --- UI & LOGIC ---

def get_sentence_to_validate(data, processed_ids):
    """Finds the next sentence to validate, prioritizing those with metaphors."""
    num_sentences = len(data["es"])
    
    # Priority 1: Sentences with metaphors in both EN and ES
    for i in range(num_sentences):
        if i not in processed_ids:
            if "B-METAPHOR" in data["es"][i]["labels"] and "B-METAPHOR" in data["en"][i]["labels"]:
                return i
    
    # Priority 2: Any remaining sentence
    for i in range(num_sentences):
        if i not in processed_ids:
            return i
            
    return None

def display_sentence(title, words, labels):
    """Displays a sentence with highlighted metaphors and labels below each word."""
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    
    word_html_parts = []
    for word, label in zip(words, labels):
        style = 'padding: 2px 5px; border-radius: 5px;' # Base style
        if label == "B-METAPHOR":
            style += ' background-color: #FFFF00; color: black;'
        
        word_html_parts.append(f'''
            <div style="display: inline-block; text-align: center; margin: 0 4px;">
                <span style="{style}">{word}</span>
                <br>
                <small style="font-size: 0.8em;">{label}</small>
            </div>
        ''')
        
    st.markdown("".join(word_html_parts), unsafe_allow_html=True)
    st.markdown("---")


st.title("Validador de Proyección de Metáforas (Catalán)")

# --- Session State Initialization ---
if 'sentence_id' not in st.session_state:
    st.session_state.sentence_id = None
if 'manual_validation' not in st.session_state:
    st.session_state.manual_validation = False
if 'correction_mode' not in st.session_state:
    st.session_state.correction_mode = False
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = ""

# --- Sidebar ---
st.sidebar.header("Selección de Dataset")
dataset_choice = st.sidebar.selectbox(
    "Elige el dataset a validar:",
    ["xnli_dev_hyp", "xnli_dev_prem", "xnli_test_hyp", "xnli_test_prem"],
    index=0
)

if dataset_choice != st.session_state.dataset_name:
    st.session_state.dataset_name = dataset_choice
    st.session_state.sentence_id = None # Reset on dataset change
    st.session_state.manual_validation = False
    st.session_state.correction_mode = False
    st.rerun()

# --- Main Page ---
if st.session_state.dataset_name:
    data = load_all_data(st.session_state.dataset_name)
    validated_ids = get_validated_ids(st.session_state.dataset_name)
    doubtful_ids = get_doubtful_ids(st.session_state.dataset_name)
    corrected_ids = get_corrected_ids(st.session_state.dataset_name)
    processed_ids = validated_ids.union(doubtful_ids).union(corrected_ids)
    
    if st.session_state.sentence_id is None:
        st.session_state.sentence_id = get_sentence_to_validate(data, processed_ids)

    sentence_id = st.session_state.sentence_id

    if sentence_id is not None:
        st.header(f"Validando Frase #{sentence_id + 1}")
        st.info(f"{len(validated_ids)} validadas, {len(doubtful_ids)} dudosas, {len(corrected_ids)} corregidas (de {len(data['es'])})")

        # Get data for the current sentence
        es_sent = data["es"][sentence_id]
        en_sent = data["en"][sentence_id]
        ca_es_proj_sent = data["ca_es_proj"][sentence_id]
        ca_en_proj_sent = data["ca_en_proj"][sentence_id]

        # Display sentences
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Castellano -> Catalán")
            display_sentence("Original (ES)", es_sent["words"], es_sent["labels"])
            display_sentence("Proyección (CA-ES)", ca_es_proj_sent["words"], ca_es_proj_sent["labels"])
        with col2:
            st.subheader("Inglés -> Catalán")
            display_sentence("Original (EN)", en_sent["words"], en_sent["labels"])
            display_sentence("Proyección (CA-EN)", ca_en_proj_sent["words"], ca_en_proj_sent["labels"])

        # --- Validation Buttons ---
        st.subheader("Acciones de Validación")
        
        projections_match = (ca_es_proj_sent["labels"] == ca_en_proj_sent["labels"])

        b_col1, b_col2, b_col3, b_col4, b_col5, b_col6 = st.columns(6)

        with b_col1:
            if st.button("✅ Validar ES-CA"):
                save_validated_sentence(st.session_state.dataset_name, sentence_id, ca_es_proj_sent["words"], ca_es_proj_sent["labels"])
                st.session_state.sentence_id = None
                st.success("Proyección ES-CA guardada.")
                st.rerun()
        
        with b_col2:
            if st.button("✅ Validar EN-CA"):
                save_validated_sentence(st.session_state.dataset_name, sentence_id, ca_en_proj_sent["words"], ca_en_proj_sent["labels"])
                st.session_state.sentence_id = None
                st.success("Proyección EN-CA guardada.")
                st.rerun()

        with b_col3:
            if st.button("🤝 Validar Ambas", disabled=not projections_match):
                save_validated_sentence(st.session_state.dataset_name, sentence_id, ca_es_proj_sent["words"], ca_es_proj_sent["labels"])
                st.session_state.sentence_id = None
                st.success("Proyección coincidente guardada.")
                st.rerun()

        with b_col4:
            if st.button("✍️ Etiquetado Manual"):
                st.session_state.manual_validation = True
                st.session_state.correction_mode = False
                st.rerun()

        with b_col5:
            if st.button("❓ Marcar como Dudosa"):
                save_doubtful_id(st.session_state.dataset_name, sentence_id)
                st.session_state.sentence_id = None
                st.warning(f"Frase #{sentence_id + 1} marcada como dudosa. No se volverá a mostrar.")
                st.rerun()
        
        with b_col6:
            if st.button("✍️ Corregir Frase (CA)"):
                st.session_state.correction_mode = True
                st.session_state.manual_validation = False
                st.rerun()


        # --- Manual Validation UI ---
        if st.session_state.manual_validation:
            st.warning("Introduce las etiquetas correctas para la frase en catalán.")
            
            words_to_validate = ca_es_proj_sent["words"] # Both projections have the same words
            
            with st.form(key="manual_form"):
                manual_labels = []
                # Create a grid for tokens and radio buttons
                cols = st.columns(len(words_to_validate))
                for i, word in enumerate(words_to_validate):
                    with cols[i]:
                        st.write(word)
                        label = st.radio(f"Label for {word}", ["O", "B-METAPHOR"], key=f"label_{i}", label_visibility="collapsed")
                        manual_labels.append(label)

                submit_button = st.form_submit_button(label='Guardar Validación Manual')

                if submit_button:
                    save_validated_sentence(st.session_state.dataset_name, sentence_id, words_to_validate, manual_labels)
                    st.session_state.sentence_id = None
                    st.session_state.manual_validation = False
                    st.success("Validación manual guardada.")
                    st.rerun()
        
        # --- Sentence Correction UI ---
        if st.session_state.correction_mode:
            st.info("Reescribe la frase en catalán y guárdala. La frase original se muestra como referencia.")
            original_sentence = " ".join(ca_es_proj_sent["words"])
            
            with st.form(key="correction_form"):
                new_sentence = st.text_area("Frase corregida:", value=original_sentence, height=100)
                
                submit_button = st.form_submit_button(label='Guardar Corrección')

                if submit_button:
                    save_retranslated_sentence(st.session_state.dataset_name, sentence_id, new_sentence)
                    st.session_state.sentence_id = None
                    st.session_state.correction_mode = False
                    st.success(f"Frase #{sentence_id + 1} corregida y guardada.")
                    st.rerun()

    else:
        st.success("¡Felicidades! Has procesado todas las frases de este dataset.")
        st.balloons()
else:
    st.info("Por favor, elige un dataset en el panel de la izquierda para empezar.")