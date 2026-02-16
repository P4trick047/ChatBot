# ## first model ##
# import streamlit as st
# from groq import Groq
# import os
# from dotenv import load_dotenv

# # ── LangChain imports for RAG
# from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings  # or use OpenAI if you want
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq

# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     st.error("GROQ_API_KEY not found. Add it to .env or Streamlit Secrets.")
#     st.stop()

# client = Groq(api_key=GROQ_API_KEY)

# # ── Available Groq models (update from https://console.groq.com/docs/models if needed)
# GROQ_MODELS = {
#     "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
#     "Llama 3.1 70B": "llama-3.1-70b-versatile",
#     "Llama 3.1 8B": "llama-3.1-8b-instant",
#     "Mixtral 8x7B": "mixtral-8x7b-32768",          # if still available
#     "Gemma 2 9B": "gemma2-9b-it",
#     # Add new ones like qwen, deepseek-r1 etc. when they appear
# }

# # ── Session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = None

# if "selected_model" not in st.session_state:
#     st.session_state.selected_model = list(GROQ_MODELS.keys())[0]

# # ── Sidebar: Model + Upload
# with st.sidebar:
#     st.header("⚙️ Settings")
    
#     selected_model_name = st.selectbox(
#         "Choose Model",
#         options=list(GROQ_MODELS.keys()),
#         index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model)
#     )
#     st.session_state.selected_model = selected_model_name
#     model_id = GROQ_MODELS[selected_model_name]

#     st.divider()
#     st.subheader("📄 Upload Documents (RAG)")
#     uploaded_files = st.file_uploader(
#         "PDF or TXT files",
#         type=["pdf", "txt"],
#         accept_multiple_files=True,
#         help="Upload documents → they become searchable context"
#     )

#     if st.button("🧹 Clear Chat & Documents"):
#         st.session_state.messages = []
#         st.session_state.vectorstore = None
#         st.rerun()

# # ── Process uploaded files (only once)
# if uploaded_files and st.session_state.vectorstore is None:
#     with st.spinner("Processing documents..."):
#         docs = []
#         for file in uploaded_files:
#             bytes_data = file.read()
#             file_name = file.name.lower()
            
#             # Save temp file (PyMuPDF / TextLoader need path or file-like)
#             with open(file_name, "wb") as f:
#                 f.write(bytes_data)

#             if file_name.endswith(".pdf"):
#                 loader = PyMuPDFLoader(file_name)
#             elif file_name.endswith(".txt"):
#                 loader = TextLoader(file_name)
#             else:
#                 st.warning(f"Skipping unsupported file: {file.name}")
#                 continue

#             docs.extend(loader.load())

#             # Clean up temp file
#             os.remove(file_name)

#         if docs:
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 length_function=len
#             )
#             splits = text_splitter.split_documents(docs)

#             # Embeddings (all-MiniLM-L6-v2 is fast & good enough)
#             embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#             st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
#             st.success(f"✅ {len(splits)} chunks indexed!")
#         else:
#             st.warning("No valid documents processed.")

# # ── Title
# st.title("Hello Bees 🐝")
# st.caption(f"Powered by Reina • {st.session_state.selected_model}")

# # ── Welcome message if empty
# if len(st.session_state.messages) == 0:
#     welcome = "Bzzzzt! 🐝 Hi I'm Reina — your beehive AI. Ask me anything… or upload documents to chat with them!"
#     st.session_state.messages.append({"role": "assistant", "content": welcome})

# # ── Display history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # ── Chat input
# if prompt := st.chat_input("Ask me anything..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""

#         try:
#             # ── Prepare context if RAG is active
#             context_str = ""
#             if st.session_state.vectorstore:
#                 retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
#                 relevant_docs = retriever.invoke(prompt)
#                 context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
#                 context_str = f"Relevant context from documents:\n{context_str}\n\nOnly use the above context if it is relevant to the question. Otherwise answer normally."

#             # System + history + context
#             system_content = f"""You are Reina, a friendly, slightly sassy bee-themed AI 🐝.
# You love subtle bee puns. Be helpful and concise.
# {context_str}"""

#             messages_for_api = [
#                 {"role": "system", "content": system_content}
#             ] + [
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ]

#             # ── Streaming with langchain-groq (cleaner streaming)
#             llm = ChatGroq(
#                 groq_api_key=GROQ_API_KEY,
#                 model_name=model_id,
#                 temperature=0.7,
#                 max_tokens=1024,
#                 streaming=True,
#             )

#             # We still use raw Groq client for compatibility with your original style
#             stream = client.chat.completions.create(
#                 messages=messages_for_api,
#                 model=model_id,
#                 temperature=0.7,
#                 max_tokens=1024,
#                 stream=True,
#             )

#             for chunk in stream:
#                 if chunk.choices[0].delta.content is not None:
#                     full_response += chunk.choices[0].delta.content
#                     message_placeholder.markdown(full_response + "▌")

#             message_placeholder.markdown(full_response)
#             st.session_state.messages.append({"role": "assistant", "content": full_response})

#         except Exception as e:
#             st.error(f"Oops... {str(e)}")

# ## 2nd model ##
# import streamlit as st
# from groq import Groq
# import os
# from dotenv import load_dotenv

# # ── LangChain imports for RAG
# from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings  # or use OpenAI if you want
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq

# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     st.error("GROQ_API_KEY not found. Add it to .env or Streamlit Secrets.")
#     st.stop()

# client = Groq(api_key=GROQ_API_KEY)

# # ── Available Groq models (update from https://console.groq.com/docs/models if needed)
# GROQ_MODELS = {
#     "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
#     "Llama 3.1 70B": "llama-3.1-70b-versatile",
#     "Llama 3.1 8B": "llama-3.1-8b-instant",
#     "Mixtral 8x7B": "mixtral-8x7b-32768",          # if still available
#     "Gemma 2 9B": "gemma2-9b-it",
#     # Add new ones like qwen, deepseek-r1 etc. when they appear
# }

# # ── Session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = None

# if "selected_model" not in st.session_state:
#     st.session_state.selected_model = list(GROQ_MODELS.keys())[0]

# # ── Sidebar: Model + Upload
# with st.sidebar:
#     st.header("⚙️ Settings")
    
#     selected_model_name = st.selectbox(
#         "Choose Model",
#         options=list(GROQ_MODELS.keys()),
#         index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model)
#     )
#     st.session_state.selected_model = selected_model_name
#     model_id = GROQ_MODELS[selected_model_name]

#     st.divider()
#     st.subheader("📄 Upload Documents (RAG)")
#     uploaded_files = st.file_uploader(
#         "PDF or TXT files",
#         type=["pdf", "txt"],
#         accept_multiple_files=True,
#         help="Upload documents → they become searchable context"
#     )

#     if st.button("🧹 Clear Chat & Documents"):
#         st.session_state.messages = []
#         st.session_state.vectorstore = None
#         st.rerun()

# # ── Process uploaded files (only once)
# if uploaded_files and st.session_state.vectorstore is None:
#     with st.spinner("Processing documents..."):
#         docs = []
#         for file in uploaded_files:
#             bytes_data = file.read()
#             file_name = file.name.lower()
            
#             # Save temp file (PyMuPDF / TextLoader need path or file-like)
#             with open(file_name, "wb") as f:
#                 f.write(bytes_data)

#             if file_name.endswith(".pdf"):
#                 loader = PyMuPDFLoader(file_name)
#             elif file_name.endswith(".txt"):
#                 loader = TextLoader(file_name)
#             else:
#                 st.warning(f"Skipping unsupported file: {file.name}")
#                 continue

#             docs.extend(loader.load())

#             # Clean up temp file
#             os.remove(file_name)

#         if docs:
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 length_function=len
#             )
#             splits = text_splitter.split_documents(docs)

#             # Embeddings (all-MiniLM-L6-v2 is fast & good enough)
#             embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#             st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
#             st.success(f"✅ {len(splits)} chunks indexed!")
#         else:
#             st.warning("No valid documents processed.")

# # ── Title
# st.title("Hello Bees 🐝")
# st.caption(f"Powered by Reina • {st.session_state.selected_model}")

# # ── Welcome message if empty
# if len(st.session_state.messages) == 0:
#     welcome = "Bzzzzt! 🐝 Hi I'm Reina — your beehive AI. Ask me anything… or upload documents to chat with them!"
#     st.session_state.messages.append({"role": "assistant", "content": welcome})

# # ── Display history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # ── Chat input
# if prompt := st.chat_input("Ask me anything..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""

#         try:
#             # ── Prepare context if RAG is active
#             context_str = ""
#             if st.session_state.vectorstore:
#                 retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
#                 relevant_docs = retriever.invoke(prompt)
#                 context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
#                 context_str = f"Relevant context from documents:\n{context_str}\n\nOnly use the above context if it is relevant to the question. Otherwise answer normally."

#             # System + history + context
#             system_content = f"""You are Reina, a friendly, slightly sassy bee-themed AI 🐝.
# You love subtle bee puns. Be helpful and concise.
# {context_str}"""

#             messages_for_api = [
#                 {"role": "system", "content": system_content}
#             ] + [
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ]

#             # ── Streaming with langchain-groq (cleaner streaming)
#             llm = ChatGroq(
#                 groq_api_key=GROQ_API_KEY,
#                 model_name=model_id,
#                 temperature=0.7,
#                 max_tokens=1024,
#                 streaming=True,
#             )

#             # We still use raw Groq client for compatibility with your original style
#             stream = client.chat.completions.create(
#                 messages=messages_for_api,
#                 model=model_id,
#                 temperature=0.7,
#                 max_tokens=1024,
#                 stream=True,
#             )

#             for chunk in stream:
#                 if chunk.choices[0].delta.content is not None:
#                     full_response += chunk.choices[0].delta.content
#                     message_placeholder.markdown(full_response + "▌")

#             message_placeholder.markdown(full_response)
#             st.session_state.messages.append({"role": "assistant", "content": full_response})

#         except Exception as e:
#             st.error(f"Oops... {str(e)}")

# ### 3rd model ###
# import streamlit as st
# from groq import Groq
# import os
# from dotenv import load_dotenv

# # LangChain / RAG imports
# from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# load_dotenv()

# # ── API Key check
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     st.error("GROQ_API_KEY not found. Please add it to .env or Streamlit Secrets.")
#     st.stop()

# client = Groq(api_key=GROQ_API_KEY)

# # ── Available models (update as needed from https://console.groq.com/docs/models)
# GROQ_MODELS = {
#     "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
#     "Llama 3.1 70B": "llama-3.1-70b-versatile",
#     "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
#     "Gemma 2 9B": "gemma2-9b-it",
#     "Mixtral 8x7B": "mixtral-8x7b-32768",
# }

# # ── Session state initialization
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = None

# if "selected_model" not in st.session_state:
#     st.session_state.selected_model = list(GROQ_MODELS.keys())[0]

# # ── Sidebar settings
# with st.sidebar:
#     st.header("⚙️ Settings")

#     # Model selection
#     selected_model_name = st.selectbox(
#         "Choose Model",
#         options=list(GROQ_MODELS.keys()),
#         index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model)
#     )
#     st.session_state.selected_model = selected_model_name
#     model_id = GROQ_MODELS[selected_model_name]

#     st.divider()

#     # Document upload for RAG
#     st.subheader("📄 Upload Documents (RAG)")
#     uploaded_files = st.file_uploader(
#         "PDF or TXT files (multiple allowed)",
#         type=["pdf", "txt"],
#         accept_multiple_files=True,
#         help="Upload files → chat with their content"
#     )

#     if st.button("🧹 Clear Chat + Documents"):
#         st.session_state.messages = []
#         st.session_state.vectorstore = None
#         st.rerun()

# # ── Process uploaded documents (only if new files and no vectorstore yet)
# if uploaded_files and st.session_state.vectorstore is None:
#     with st.spinner("Indexing documents... 🐝"):
#         all_docs = []
#         for uploaded_file in uploaded_files:
#             file_bytes = uploaded_file.read()
#             file_name = uploaded_file.name.lower()

#             # Temporary file for loaders
#             temp_path = f"temp_{file_name}"
#             with open(temp_path, "wb") as f:
#                 f.write(file_bytes)

#             try:
#                 if file_name.endswith(".pdf"):
#                     loader = PyMuPDFLoader(temp_path)
#                 elif file_name.endswith(".txt"):
#                     loader = TextLoader(temp_path)
#                 else:
#                     st.warning(f"Skipped unsupported file: {uploaded_file.name}")
#                     continue

#                 all_docs.extend(loader.load())
#             finally:
#                 if os.path.exists(temp_path):
#                     os.remove(temp_path)

#         if all_docs:
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200
#             )
#             chunks = text_splitter.split_documents(all_docs)

#             embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#             st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
#             st.success(f"✅ Ready! {len(chunks)} chunks from {len(uploaded_files)} file(s)")
#         else:
#             st.warning("No valid content extracted from uploaded files.")

# # ── Title & caption
# st.title("Hello Bees 🐝")
# st.caption(f"Powered by Reina • {st.session_state.selected_model}")

# # ── Welcome message (only first time)
# if len(st.session_state.messages) == 0:
#     welcome = (
#         "Bzzzzt! 🐝 Hi I'm Reina — your beehive buddy!\n\n"
#         "Ask me anything... or upload documents to chat about them.\n"
#         "You can also use the 🎤 microphone icon to speak your question!"
#     )
#     st.session_state.messages.append({"role": "assistant", "content": welcome})

# # ── Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # ── Chat input with voice support
# if prompt := st.chat_input(
#     "Ask me anything... (type or use 🎤 mic)",
#     accept_audio=True,
#     max_chars=4000
# ):
#     # Handle text vs voice input
#     if isinstance(prompt, dict):
#         # Voice input
#         transcribed_text = prompt.get("text", "").strip()
#         audio_bytes = prompt.get("audio")

#         if not transcribed_text:
#             st.warning("Didn't catch that — try speaking again?")
#             st.rerun()

#         user_content = transcribed_text
#         display_content = transcribed_text
#         is_voice = True
#     else:
#         # Text input
#         user_content = prompt
#         display_content = prompt
#         audio_bytes = None
#         is_voice = False

#     # Add to history & display
#     st.session_state.messages.append({"role": "user", "content": user_content})
#     with st.chat_message("user"):
#         st.markdown(display_content)
#         if is_voice and audio_bytes:
#             st.caption("🎤 Voice message")
#             # Optional: let user replay what they said
#             st.audio(audio_bytes, format="audio/wav")

#     # ── Generate response
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""

#         try:
#             # Retrieve context if RAG is active
#             context_parts = []
#             if st.session_state.vectorstore:
#                 retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
#                 relevant_docs = retriever.invoke(user_content)
#                 context_parts = [doc.page_content for doc in relevant_docs]

#             context_str = ""
#             if context_parts:
#                 context_str = "\n\n".join(context_parts)
#                 context_str = f"""Relevant information from uploaded documents:
# {context_str}

# Use this context only if it helps answer the question. Otherwise respond normally."""

#             # System prompt (personality + context)
#             system_prompt = f"""You are Reina, a friendly, slightly sassy bee-themed AI assistant 🐝.
# You enjoy subtle bee puns and emojis, but don't overdo it.
# Be helpful, concise, and engaging.
# {context_str}"""

#             # Build messages for Groq
#             api_messages = [
#                 {"role": "system", "content": system_prompt}
#             ] + [
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ]

#             # Streaming call
#             stream = client.chat.completions.create(
#                 messages=api_messages,
#                 model=model_id,
#                 temperature=0.7,
#                 max_tokens=1200,
#                 stream=True,
#             )

#             for chunk in stream:
#                 if chunk.choices[0].delta.content is not None:
#                     full_response += chunk.choices[0].delta.content
#                     message_placeholder.markdown(full_response + "▌")

#             message_placeholder.markdown(full_response)
#             st.session_state.messages.append({"role": "assistant", "content": full_response})

#         except Exception as e:
#             st.error(f"Oops, hive malfunction: {str(e)}")

### 4th model ###
import streamlit as st
from groq import Groq
import os
import pandas as pd
from dotenv import load_dotenv

# LangChain / RAG imports (same as before)
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── NEW: Google Sheets connection
from streamlit_gsheets import GSheetsConnection

load_dotenv()

# ── API Key check (unchanged)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Models dict (unchanged)
GROQ_MODELS = {
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "Gemma 2 9B": "gemma2-9b-it",
}

# ── Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(GROQ_MODELS.keys())[0]

# ── Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Model
    selected_model_name = st.selectbox("Model", list(GROQ_MODELS.keys()))
    st.session_state.selected_model = selected_model_name
    model_id = GROQ_MODELS[selected_model_name]

    st.divider()

    # ── Google Sheet input
    st.subheader("📊 Google Sheet (RAG)")
    sheet_url = st.text_input("Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...")
    worksheet_name = st.text_input("Worksheet name / gid", value="Sheet1")
    if st.button("🔄 Load & Index Sheet"):
        if sheet_url:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection, url=sheet_url)
                df = conn.read(worksheet=worksheet_name)
                st.session_state["sheet_df"] = df
                st.success(f"Loaded {len(df)} rows from sheet!")
            except Exception as e:
                st.error(f"Cannot read sheet: {e}\nCheck URL, sharing settings or secrets.toml")
        else:
            st.warning("Enter sheet URL first")

    # Existing file upload (PDF/TXT)
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader("PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)

    if st.button("🧹 Clear Everything"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        if "sheet_df" in st.session_state:
            del st.session_state["sheet_df"]
        st.rerun()

# ── Index documents + sheet (runs when needed)
if st.session_state.vectorstore is None:
    docs = []

    # ── Add sheet rows as Documents (if loaded)
    if "sheet_df" in st.session_state and not st.session_state.sheet_df.empty:
        df = st.session_state.sheet_df.astype(str)
        # Simple way: one document per row
        for i, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if val.strip()])
            if row_text.strip():
                docs.append(Document(
                    page_content=row_text,
                    metadata={"source": "google_sheet", "row": i}
                ))
        st.info(f"Indexed {len(docs)} rows from Google Sheet")

    # ── Add uploaded files (your existing code, slightly refactored)
    if uploaded_files:
        with st.spinner("Processing files..."):
            for file in uploaded_files:
                file_bytes = file.read()
                file_name = file.name.lower()
                temp_path = f"temp_{file_name}"
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)

                try:
                    if file_name.endswith(".pdf"):
                        loader = PyMuPDFLoader(temp_path)
                    elif file_name.endswith(".txt"):
                        loader = TextLoader(temp_path)
                    else:
                        continue
                    file_docs = loader.load()
                    for d in file_docs:
                        d.metadata["source"] = file.name
                    docs.extend(file_docs)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    if docs:
        with st.spinner("Creating vector index..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.success(f"Vector store ready with {len(chunks)} chunks (sheet + files)")

# ── Title & welcome (unchanged, but updated welcome text)
st.title("Hello Bees 🐝")
st.caption(f"Powered by Reina • {st.session_state.selected_model}")

if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Bzzzzt! 🐝 Hi! I'm Reina.\nI can now talk to your **Google Sheet** + uploaded files + internet knowledge.\nPaste your sheet URL in sidebar & click Load → ask anything!"
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat + voice (unchanged)
if prompt := st.chat_input("Ask about sheet, files, or anything...", accept_audio=True):
    if isinstance(prompt, dict):
        user_content = prompt.get("text", "").strip()
        audio_bytes = prompt.get("audio")
        if not user_content:
            st.warning("No speech recognized.")
            st.rerun()
        display_content = user_content
        is_voice = True
    else:
        user_content = prompt
        display_content = prompt
        audio_bytes = None
        is_voice = False

    st.session_state.messages.append({"role": "user", "content": user_content})
    with st.chat_message("user"):
        st.markdown(display_content)
        if is_voice and audio_bytes:
            st.caption("🎤 Voice input")
            st.audio(audio_bytes, format="audio/wav")

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            context_str = ""
            if st.session_state.vectorstore:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                relevant = retriever.invoke(user_content)
                context_parts = [doc.page_content for doc in relevant]
                if context_parts:
                    context_str = "\n\n".join(context_parts)
                    context_str = f"""Relevant data from Google Sheet / documents:\n{context_str}\n\nUse ONLY if relevant. Otherwise answer normally without mentioning 'documents' or 'sheet' unless asked."""

            system_prompt = f"""You are Reina, friendly bee-themed AI 🐝.
Slightly sassy, helpful, concise.
Use subtle bee puns sometimes.
{context_str}"""

            api_messages = [{"role": "system", "content": system_prompt}] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            stream = client.chat.completions.create(
                messages=api_messages,
                model=model_id,
                temperature=0.7,
                max_tokens=1200,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Hive error: {e}")
