import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

# ── LangChain imports for RAG
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # or use OpenAI if you want
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to .env or Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ── Available Groq models (update from https://console.groq.com/docs/models if needed)
GROQ_MODELS = {
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Mixtral 8x7B": "mixtral-8x7b-32768",          # if still available
    "Gemma 2 9B": "gemma2-9b-it",
    # Add new ones like qwen, deepseek-r1 etc. when they appear
}

# ── Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(GROQ_MODELS.keys())[0]

# ── Sidebar: Model + Upload
with st.sidebar:
    st.header("⚙️ Settings")
    
    selected_model_name = st.selectbox(
        "Choose Model",
        options=list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model_name
    model_id = GROQ_MODELS[selected_model_name]

    st.divider()
    st.subheader("📄 Upload Documents (RAG)")
    uploaded_files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents → they become searchable context"
    )

    if st.button("🧹 Clear Chat & Documents"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.rerun()

# ── Process uploaded files (only once)
if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("Processing documents..."):
        docs = []
        for file in uploaded_files:
            bytes_data = file.read()
            file_name = file.name.lower()
            
            # Save temp file (PyMuPDF / TextLoader need path or file-like)
            with open(file_name, "wb") as f:
                f.write(bytes_data)

            if file_name.endswith(".pdf"):
                loader = PyMuPDFLoader(file_name)
            elif file_name.endswith(".txt"):
                loader = TextLoader(file_name)
            else:
                st.warning(f"Skipping unsupported file: {file.name}")
                continue

            docs.extend(loader.load())

            # Clean up temp file
            os.remove(file_name)

        if docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(docs)

            # Embeddings (all-MiniLM-L6-v2 is fast & good enough)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            st.success(f"✅ {len(splits)} chunks indexed!")
        else:
            st.warning("No valid documents processed.")

# ── Title
st.title("Hello Bees 🐝")
st.caption(f"Powered by Reina • {st.session_state.selected_model}")

# ── Welcome message if empty
if len(st.session_state.messages) == 0:
    welcome = "Bzzzzt! 🐝 Hi I'm Reina — your beehive AI. Ask me anything… or upload documents to chat with them!"
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# ── Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # ── Prepare context if RAG is active
            context_str = ""
            if st.session_state.vectorstore:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                relevant_docs = retriever.invoke(prompt)
                context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
                context_str = f"Relevant context from documents:\n{context_str}\n\nOnly use the above context if it is relevant to the question. Otherwise answer normally."

            # System + history + context
            system_content = f"""You are Reina, a friendly, slightly sassy bee-themed AI 🐝.
You love subtle bee puns. Be helpful and concise.
{context_str}"""

            messages_for_api = [
                {"role": "system", "content": system_content}
            ] + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            # ── Streaming with langchain-groq (cleaner streaming)
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=model_id,
                temperature=0.7,
                max_tokens=1024,
                streaming=True,
            )

            # We still use raw Groq client for compatibility with your original style
            stream = client.chat.completions.create(
                messages=messages_for_api,
                model=model_id,
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
            st.error(f"Oops... {str(e)}")
