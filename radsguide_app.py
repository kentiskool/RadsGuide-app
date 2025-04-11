import streamlit as st
import openai
import pandas as pd

# Use API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.write("✅ CSV loaded successfully!")

# Load your imaging decision support database
df = pd.read_csv("test.csv", skiprows=1)  # skips duplicate header row from export

# Configure Streamlit page
st.set_page_config(page_title="RadsGuide Chatbot", layout="centered")
st.title("🧠 RadsGuide: Imaging Decision Assistant")
st.markdown("""
Type a clinical question like:

- "How do I evaluate for appendicitis in pregnancy?"
- "What scan do I need for suspected PE?"
- "Patient has sudden severe headache—what do I order?"

RadsGuide will recommend the most appropriate imaging study based on your dataset.
""")

# Input box
user_input = st.text_input("Enter your clinical question:", placeholder="e.g., Rule out pneumonia in immunocompromised patient")

if user_input:
    with st.spinner("Thinking..."):
        # Create GPT prompt
        prompt = f"""
        You are a radiology assistant. Match the clinical question below to the most appropriate entry
        from this list of clinical indications:

        {df['Clinical indication'].dropna().to_list()}

        Clinical question: \"{user_input}\"

        Respond ONLY with the best-matching phrase from the list.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful radiology assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            match_phrase = response.choices[0].message["content"].strip()

            # Search your dataframe for a match
            match_row = df[df['Clinical indication'].str.strip() == match_phrase]

            if not match_row.empty:
                row = match_row.iloc[0]
                st.success("✅ Imaging Recommendation:")
                st.markdown(f"**Clinical Indication:** {match_phrase}")
                st.markdown(f"**Modality:** {row['Modality']}")
                st.markdown(f"**Contrast:** {row['Contrast']}")
                st.markdown(f"**ACR Score:** {row['ACR Score']}")
                st.markdown(f"**Notes:** {row['Notes/Caveats']}")
                st.markdown(f"**Special Populations:** {row['Special population'] if pd.notna(row['Special population']) else 'None'}")
                st.markdown(f"**Logic Notes:** {row['Logic Notes'] if pd.notna(row['Logic Notes']) else 'None'}")
                st.markdown(f"[📄 View ACR Guidelines]({row['ACR link']})")
            else:
                st.warning("No exact match found. Try rephrasing or simplifying the question.")

        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Built by Kent Kleinschmidt · Powered by OpenAI and RadsGuide")
