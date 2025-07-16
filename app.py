import streamlit as st
import pandas as pd
import openai
import os
import numpy as np
import faiss
from typing import List

# --- Simple password protection ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "slu123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # Disclaimer removed from password prompt

    if "password_correct" not in st.session_state:
        st.text_input("Enter password to access RadsGuide:", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password to access RadsGuide:", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        st.stop()

check_password()

# Load OpenAI API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=api_key)

# Load the dataset
@st.cache_data

def load_data():
    df = pd.read_csv('RadsGuideData_clean.csv')
    df['canonical'] = df['canonical'].astype(str)
    df['synonyms'] = df['synonyms'].astype(str)
    df['Modality'] = df['Modality'].astype(str)
    df['Clinical Indication'] = df['Clinical Indication'].astype(str)
    return df

data = load_data()

# Prepare all phrases (canonical + synonyms)
all_phrases = []
phrase_to_row = []  # Maps phrase index to row index in data
for idx, row in data.iterrows():
    # Add canonical
    all_phrases.append(str(row['canonical']))
    phrase_to_row.append(idx)
    # Add synonyms (split by semicolon or comma)
    syns = str(row['synonyms'])
    if syns and syns.lower() != 'nan':
        for syn in syns.replace(';', ',').split(','):
            syn = syn.strip()
            if syn:
                all_phrases.append(syn)
                phrase_to_row.append(idx)

# --- Embedding utilities ---
@st.cache_data(show_spinner=True)
def get_phrase_embeddings(phrases: List[str]):
    # Get embeddings for all phrases (batch)
    response = client.embeddings.create(
        input=phrases,
        model="text-embedding-3-small"
    )
    return np.array([e.embedding for e in response.data], dtype=np.float32)

@st.cache_data(show_spinner=False)
def get_query_embedding(query: str):
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# Build FAISS index
phrase_embeddings = get_phrase_embeddings(all_phrases)
index = faiss.IndexFlatL2(phrase_embeddings.shape[1])
index.add(phrase_embeddings)

# Streamlit UI
st.title('RadsGuide: ER Imaging Recommendation Chatbot')
st.markdown('**Based on ACR Appropriateness Criteria tailored to the adult ED setting and SLU-specific protocols**')
st.write('Ask for an imaging recommendation (e.g., "rule out PE", "appendicitis").')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for msg in st.session_state['messages']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# User input
user_input = st.chat_input('Enter your clinical question...')

# Abbreviation expansion dictionary for common clinical terms
ABBREVIATION_MAP = {
    "ruq": "right upper quadrant",
    "luq": "left upper quadrant",
    "rlq": "right lower quadrant",
    "llq": "left lower quadrant",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "uti": "urinary tract infection",
    "gi": "gastrointestinal",
    "ct": "computed tomography",
    "us": "ultrasound",
    "fx": "fracture",
    "abd": "abdomen",
    "hx": "history",
    "w/u": "workup",
    "r/o": "rule out",
    "c/o": "complains of",
    "n/v": "nausea and vomiting",
    "loc": "loss of consciousness",
    "s/p": "status post",
    "h/o": "history of",
    "wbc": "white blood cell",
    "iv": "intravenous",
    # Add more as needed
}

def expand_abbreviations(text):
    words = text.split()
    expanded = []
    for word in words:
        key = word.lower().strip('.,;:')
        if key in ABBREVIATION_MAP:
            expanded.append(ABBREVIATION_MAP[key])
        else:
            expanded.append(word)
    return ' '.join(expanded)

if user_input:
    # Preprocess user input to expand abbreviations
    expanded_input = expand_abbreviations(user_input)
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # Get embedding for user query
    query_embedding = get_query_embedding(expanded_input).reshape(1, -1).astype(np.float32)
    # Determine if the query is asking for next imaging after a negative result
    next_imaging_keywords = [
        'negative ultrasound',
        'negative ct',
        'negative initial imaging',
        'negative xray',
        'negative radiograph',
        'negative mri',
        'negative scan',
        'negative study',
        'next step',
        'next imaging',
        'follow-up imaging',
        'f/u imaging',
    ]
    is_next_imaging = any(kw in expanded_input.lower() for kw in next_imaging_keywords)

    # Search FAISS index for top 5 best matches
    D, I = index.search(query_embedding, 5)
    best_idx = int(I[0][0])
    best_distance = float(D[0][0])
    # Try to prioritize a match with 'next imaging' or 'initial imaging' in the phrase
    prioritized_idx = None
    for rank in range(I.shape[1]):
        phrase = all_phrases[I[0][rank]].lower()
        if is_next_imaging and 'next imaging' in phrase:
            prioritized_idx = rank
            break
        elif not is_next_imaging and 'initial imaging' in phrase:
            prioritized_idx = rank
            break
    if prioritized_idx is not None:
        best_idx = int(I[0][prioritized_idx])
        best_distance = float(D[0][prioritized_idx])
    row_idx = phrase_to_row[best_idx]
    matched_row = data.iloc[row_idx]
    matched_phrase = all_phrases[best_idx]
    modality = matched_row['Modality']
    clinical_indication = matched_row['Clinical Indication']

    # Debug display: show best match and distance
    st.info(f"**DEBUG:** Best match: '{matched_phrase}' (distance: {best_distance:.4f})")

    # Set a similarity threshold (lower distance = better match)
    SIMILARITY_THRESHOLD = 0.5  # Raised threshold for more flexible matching

    # Escalation terms for more advanced imaging
    escalation_terms = [
        'negative', 'fever', 'elevated wbc', 'next step', 'next imaging', 'follow-up', 'f/u', 'complicated', 'abnormal', 'equivocal', 'persistent', 'worsening', 'not improved', 'no improvement', 'unchanged', 'refractory', 'unresolved', 'recurring', 'repeat', 'second', 'subsequent', 'after', 'post', 'failed', 'inconclusive', 'indeterminate', 'unclear', 'cannot exclude', 'cannot rule out'
    ]
    is_simple_query = not any(term in expanded_input.lower() for term in escalation_terms)

    # Show top 3 matches if they are within 20% of the best distance, else just the best match
    margin = 0.2  # 20% margin
    top_matches = []
    initial_imaging_idx = None
    for rank in range(min(3, I.shape[1])):
        idx = int(I[0][rank])
        dist = float(D[0][rank])
        row_idx = phrase_to_row[idx]
        matched_row = data.iloc[row_idx]
        matched_phrase = all_phrases[idx]
        modality = matched_row['Modality']
        clinical_indication = matched_row['Clinical Indication']
        top_matches.append((modality, clinical_indication, matched_phrase, dist))
        if 'initial imaging' in matched_phrase.lower() and initial_imaging_idx is None:
            initial_imaging_idx = len(top_matches) - 1

    # For simple queries, if one of the top 3 matches is 'initial imaging', move it to the top
    if is_simple_query and initial_imaging_idx is not None and len(top_matches) > 1:
        # Move the initial imaging match to the front
        initial_match = top_matches.pop(initial_imaging_idx)
        top_matches = [initial_match] + top_matches

    # Deduplicate top_matches by (clinical_indication, modality)
    seen = set()
    deduped_matches = []
    for match in top_matches:
        key = (match[1], match[0])  # (clinical_indication, modality)
        if key not in seen:
            deduped_matches.append(match)
            seen.add(key)
        if len(deduped_matches) == 3:
            break
    top_matches = deduped_matches

    acr_reference = '\n\n_For more information, see the [ACR Appropriateness Criteria](https://gravitas.acr.org/acportal)._'  # Reference line

    if len(top_matches) == 1:
        modality, clinical_indication, matched_phrase, dist = top_matches[0]
        answer = f"**Recommended imaging modality:** {modality}\n\n_Clinical indication matched:_\n{clinical_indication}" + acr_reference
    elif len(top_matches) > 1:
        answer = "**Top relevant imaging recommendations:**\n"
        for i, (modality, clinical_indication, matched_phrase, dist) in enumerate(top_matches, 1):
            answer += f"\n**Option {i}:** {modality}\n\n_Clinical indication matched:_\n{clinical_indication}\n"
        answer += acr_reference
    else:
        # Fallback: Use GPT-4o-mini to generate a response referencing the ACR site
        gpt_prompt = (
            f"A clinician is seeking the most appropriate imaging modality for the following clinical scenario: '{user_input}'. "
            "You do not have a local dataset to reference. Based on your knowledge and the ACR Appropriateness Criteria (https://www.acr.org/Clinical-Resources/ACR-Appropriateness-Criteria), what is the best recommendation? "
            "Clearly state that this answer is based on your model knowledge and not a live web search."
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical imaging recommendation assistant."},
                {"role": "user", "content": gpt_prompt}
            ],
            max_tokens=300,
            temperature=0.2
        )
        gpt_answer = response.choices[0].message.content if response.choices[0].message.content else "No answer returned."
        gpt_answer = gpt_answer.strip()
        answer = (
            f"**No close match found in the local dataset.**\n\n"
            f"**GPT-4o-mini result:**\n{gpt_answer}\n\n"
            f"_This answer was generated by GPT-4o-mini based on its model knowledge and the ACR Appropriateness Criteria, not from a live web search or the local dataset. For more information, see the [ACR Appropriateness Criteria](https://www.acr.org/Clinical-Resources/ACR-Appropriateness-Criteria)._"
        )

    st.session_state['messages'].append({'role': 'assistant', 'content': answer})
    with st.chat_message('assistant'):
        st.markdown(answer) 

# Disclaimer (always visible)
st.markdown(
    """
    <hr>
    <sub>Disclaimer: This tool is for informational purposes only and does not constitute medical advice. Always consult institutional protocols and clinical judgment.</sub>
    """,
    unsafe_allow_html=True
) 