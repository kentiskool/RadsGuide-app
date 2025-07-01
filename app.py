import streamlit as st
import pandas as pd
import openai
import os

# Load OpenAI API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=api_key)

# Load the dataset
@st.cache_data

def load_data():
    df = pd.read_csv('RadsGuideDataCSV.csv')
    df['clinical indication'] = df['clinical indication'].astype(str)
    df['modality'] = df['modality'].astype(str)
    return df

data = load_data()

# Build a list of all clinical indications
indications = data['clinical indication'].tolist()

# --- Simple password protection ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "slu123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password to access RadsGuide:", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password to access RadsGuide:", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        st.stop()

check_password()

# Streamlit UI
st.title('RadsGuide: ER Imaging Recommendation Chatbot for SLU')
st.markdown('Based on ACR Appropriateness Criteria, tailored to the adult ED setting and SLU-specific protocols')
st.write('Ask for an imaging recommendation (e.g., "rule out PE", "abdominal pain", "appendicitis").')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for msg in st.session_state['messages']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# User input
user_input = st.chat_input('Enter your clinical question...')

if user_input:
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # Build the system prompt
    system_prompt = (
        "You are a medical imaging recommendation assistant. "
        "Given a clinical question, select the single closest matching clinical indication from the list below. "
        "Only respond with the exact clinical indication from the list. Do not invent or extrapolate.\n\n"
        f"Clinical indications:\n- " + '\n- '.join(indications)
    )

    # Call OpenAI to map input to indication (new API syntax)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        max_tokens=50,
        temperature=0
    )
    mapped_indication = None
    if response.choices and response.choices[0].message and response.choices[0].message.content:
        mapped_indication = response.choices[0].message.content.strip()
    else:
        mapped_indication = ""

    # Find the modality
    modality = data.loc[data['clinical indication'] == mapped_indication, 'modality'].values
    acr_reference = '\n\n_For more information, see the [ACR Appropriateness Criteria](https://gravitas.acr.org/acportal)._'  # Reference line
    if len(modality) > 0:
        answer = f"**Recommended imaging modality:** {modality[0]}\n\n_Clinical indication matched: {mapped_indication}_" + acr_reference
    else:
        answer = "Sorry, I couldn't find a matching clinical indication." + acr_reference

    st.session_state['messages'].append({'role': 'assistant', 'content': answer})
    with st.chat_message('assistant'):
        st.markdown(answer) 