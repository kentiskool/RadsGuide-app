import streamlit as st
import openai
import pandas as pd

# Use your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load the clean imaging dataset
df = pd.read_csv("RadsGuideData.csv")

# App config
st.set_page_config(page_title="RadsGuide Chatbot", layout="centered")
st.title("🧠 RadsGuide: Imaging Decision Assistant")

st.markdown("""
Type a clinical question like:

- "Suspected PE in a pregnant patient"
- "RUQ pain in pregnancy"
- "Sudden headache"

RadsGuide will find the most appropriate imaging recommendation.
""")

# Input field
user_input = st.text_input("Enter your clinical question:", placeholder="e.g., Rule out pulmonary embolism")

if user_input:
    with st.spinner("Thinking..."):
        try:
            # GPT prompt
            prompt = f"""
            You are a radiology assistant. Match the clinical question below to the most appropriate entry
            from this list of clinical indications:

            {df['Clinical indication'].dropna().to_list()}

            Clinical question: \"{user_input}\"

            Respond ONLY with the best-matching phrase from the list.
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful radiology assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            match_phrase = response.choices[0].message["content"].strip()

            match_row = df[df['Clinical indication'].str.strip() == match_phrase]

            if not match_row.empty:
                row = match_row.iloc[0]
                st.success("✅ Imaging Recommendation:")
                st.markdown(f"**Clinical Indication:** {match_phrase}")
                st.markdown(f"**Modality:** {row['Modality']}")
                st.markdown(f"**Contrast:** {row['Contrast']}")
                st.markdown(f"**ACR Score:** {row['ACR Score']}")
                st.markdown(f"**Notes:** {row['Notes/Caveats']}")
                st.markdown(f"**Special Population:** {row['Special population'] if pd.notna(row['Special population']) else 'None'}")
                st.markdown(f"**Logic Notes:** {row['Logic Notes'] if pd.notna(row['Logic Notes']) else 'None'}")
                st.markdown(f"**Category:** {row['Category']}")
                st.markdown(f"**Last Reviewed:** {row['Last reviewed']}")
            else:
                st.warning("No exact match found. Try rephrasing your question.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("Built by Kent Kleinschmidt · Powered by OpenAI and RadsGuide")
