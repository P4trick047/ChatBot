import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()           # loads .env file (create it locally, don't commit!)

# ── Get API key (never hardcode it!)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to Secrets in Streamlit Cloud or .env locally.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ── Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Title & description
st.title("My Free AI Chatbot 🚀")
st.caption("Powered by Reina")

# ── Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── User input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=1024,
                    stream=True,
                    )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {e}")
