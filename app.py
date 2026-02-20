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

# ### 4th model (connect with google sheet) ###
# import streamlit as st
# from groq import Groq
# import os
# from dotenv import load_dotenv
# import requests
# import pandas as pd
# import io

# # ── LangChain imports for RAG
# from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
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
#     "Mixtral 8x7B": "mixtral-8x7b-32768",
#     "Gemma 2 9B": "gemma2-9b-it",
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

# # ── Google Sheet public CSV loader
# @st.cache_data(ttl=1800)  # 30 minutes cache
# def load_google_sheet_csv(spreadsheet_id, gid=0):
#     export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
#     try:
#         response = requests.get(export_url, timeout=10)
#         response.raise_for_status()
#         csv_data = io.StringIO(response.text)
#         df = pd.read_csv(csv_data)
#         return df
#     except Exception as e:
#         st.warning(f"Could not load Google Sheet: {str(e)}\n"
#                    "Make sure the sheet is shared as 'Anyone with the link → Viewer'")
#         return pd.DataFrame()

# SHEET_ID = "1ATllEOsVzBIHm4egctEVbf7CDzmHtFfyEMmT7U6NNnw"
# GID = 0  # change if you want a different tab

# # ── Process uploaded files (only once)
# if uploaded_files and st.session_state.vectorstore is None:
#     with st.spinner("Processing documents..."):
#         docs = []
#         for file in uploaded_files:
#             bytes_data = file.read()
#             file_name = file.name.lower()
            
#             # Save temp file
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
#             os.remove(file_name)
        
#         if docs:
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 length_function=len
#             )
#             splits = text_splitter.split_documents(docs)
            
#             embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#             st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
#             st.success(f"✅ {len(splits)} chunks indexed!")
#         else:
#             st.warning("No valid documents processed.")

# # ── Title
# st.title("Hello Bees 🐝")
# st.caption(f"Powered by Reina • {st.session_state.selected_model}")

# # ── Welcome message
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
#             # ── Prepare context
#             context_str = ""
            
#             # RAG from uploaded docs
#             if st.session_state.vectorstore:
#                 retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
#                 relevant_docs = retriever.invoke(prompt)
#                 doc_text = "\n\n".join([doc.page_content for doc in relevant_docs])
#                 if doc_text:
#                     context_str += f"Relevant context from uploaded documents:\n{doc_text}\n\n"
            
#             # Google Sheet data
#             df_sheet = load_google_sheet_csv(SHEET_ID, gid=GID)
#             if not df_sheet.empty:
#                 # Choose format that works best for your sheet
#                 sheet_text = df_sheet.to_markdown(index=False)
#                 # sheet_text = df_sheet.head(30).to_string(index=False)  # alternative
#                 context_str += (
#                     f"Current data from Google Sheet (bee/hive records):\n"
#                     f"{sheet_text}\n\n"
#                     "Use this data to answer questions about records, dates, amounts, "
#                     "production, hives, inventory, or any numbers/names/dates shown. "
#                     "Prioritize the most recent entries if dates exist."
#                 )
#             else:
#                 context_str += "(Google Sheet data currently unavailable)\n"
            
#             # System prompt
#             system_content = f"""You are Reina, a friendly, slightly sassy bee-themed AI 🐝.
# You love subtle bee puns. Be helpful, concise, and engaging.
# {context_str}"""

#             messages_for_api = [
#                 {"role": "system", "content": system_content}
#             ] + [
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ]
            
#             # ── Streaming response
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
# ### Updated code with Gemini ### 
# import streamlit as st
# import os
# from dotenv import load_dotenv
# import requests
# import pandas as pd
# import io
# from google import genai

# # ── LangChain imports
# from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # ──────────────────────────────
# # ENV
# # ──────────────────────────────
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# if not GOOGLE_API_KEY:
#     st.error("GOOGLE_API_KEY not found.")
#     st.stop()

# client = genai.Client(api_key=GOOGLE_API_KEY)

# # ──────────────────────────────
# # AVAILABLE MODELS (SAFE ONES)
# # ──────────────────────────────
# GEMINI_MODELS = {
#     "Gemini 1.5 Flash": "gemini-1.5-flash",
#     "Gemini 1.5 Pro": "gemini-1.5-pro",
# }

# # ──────────────────────────────
# # SESSION
# # ──────────────────────────────
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = None

# if "selected_model" not in st.session_state:
#     st.session_state.selected_model = list(GEMINI_MODELS.keys())[0]

# # ──────────────────────────────
# # SIDEBAR
# # ──────────────────────────────
# with st.sidebar:
#     st.header("⚙️ Settings")

#     selected_model_name = st.selectbox(
#         "Choose Model",
#         list(GEMINI_MODELS.keys())
#     )

#     st.session_state.selected_model = selected_model_name
#     model_id = GEMINI_MODELS[selected_model_name]

#     st.divider()

#     uploaded_files = st.file_uploader(
#         "Upload PDF or TXT (RAG)",
#         type=["pdf", "txt"],
#         accept_multiple_files=True
#     )

#     if st.button("Clear Chat"):
#         st.session_state.messages = []
#         st.session_state.vectorstore = None
#         st.rerun()

# # ──────────────────────────────
# # GOOGLE SHEET
# # ──────────────────────────────
# @st.cache_data(ttl=1800)
# def load_google_sheet_csv(spreadsheet_id, gid=0):
#     url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
#     try:
#         response = requests.get(url)
#         df = pd.read_csv(io.StringIO(response.text))
#         return df
#     except:
#         return pd.DataFrame()

# SHEET_ID = "1ATllEOsVzBIHm4egctEVbf7CDzmHtFfyEMmT7U6NNnw"
# GID = 0

# # ──────────────────────────────
# # RAG PROCESSING
# # ──────────────────────────────
# if uploaded_files and st.session_state.vectorstore is None:
#     docs = []

#     for file in uploaded_files:
#         bytes_data = file.read()
#         filename = file.name.lower()

#         with open(filename, "wb") as f:
#             f.write(bytes_data)

#         if filename.endswith(".pdf"):
#             loader = PyMuPDFLoader(filename)
#         elif filename.endswith(".txt"):
#             loader = TextLoader(filename)
#         else:
#             continue

#         docs.extend(loader.load())
#         os.remove(filename)

#     if docs:
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         splits = splitter.split_documents(docs)

#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )

#         st.session_state.vectorstore = FAISS.from_documents(
#             splits,
#             embeddings
#         )

#         st.success("Documents indexed!")

# # ──────────────────────────────
# # UI
# # ──────────────────────────────
# st.title("Hello Bees 🐝")
# st.caption(f"Powered by Gemini • {st.session_state.selected_model}")

# if len(st.session_state.messages) == 0:
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": "Hi! I'm Reina 🐝 Ask me anything."
#     })

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # ──────────────────────────────
# # CHAT
# # ──────────────────────────────
# if prompt := st.chat_input("Ask me anything..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         placeholder = st.empty()
#         full_response = ""

#         try:
#             context = ""

#             # RAG
#             if st.session_state.vectorstore:
#                 retriever = st.session_state.vectorstore.as_retriever()
#                 docs = retriever.invoke(prompt)
#                 context += "\n\n".join([d.page_content for d in docs])

#             # Google Sheet
#             df_sheet = load_google_sheet_csv(SHEET_ID, GID)
#             if not df_sheet.empty:
#                 context += "\n\nGoogle Sheet Data:\n"
#                 context += df_sheet.to_string(index=False)

#             final_prompt = f"""
# You are Reina, a helpful bee-themed AI.

# Context:
# {context}

# User Question:
# {prompt}
# """

#             response = client.models.generate_content_stream(
#                 model=model_id,
#                 contents=final_prompt,
#             )

#             for chunk in response:
#                 if chunk.text:
#                     full_response += chunk.text
#                     placeholder.markdown(full_response + "▌")

#             placeholder.markdown(full_response)

#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "content": full_response
#             })

#         except Exception as e:
#             st.error(f"Error: {str(e)}")
### interactive UI in model 4h ####
import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
import requests
import pandas as pd
import io

# ── LangChain imports for RAG
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from streamlit_emoji_float import emoji_float
    FLOAT_EMOJIS_AVAILABLE = True
except ImportError:
    FLOAT_EMOJIS_AVAILABLE = False

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to .env or Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

GROQ_MODELS = {
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it",
}

# ── Page config
st.set_page_config(
    page_title="Reina 🐝 Beehive AI",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Floating emojis (optional – install streamlit-emoji-float if you want)
if FLOAT_EMOJIS_AVAILABLE:
    emoji_float(
        emojis=["🐝", "🐝", "🍯", "🌼", "💛"],
        count=12,
        minSize=22,
        maxSize=50,
        duration=16.0,
        fadeOut=True,
        opacity=0.65
    )

# ── Interactive Honeycomb Background + Bee Animations ────────────────────────
st.components.v1.html("""
<div class="honeycomb-bg"></div>

<style>
  .honeycomb-bg {
    position: fixed;
    inset: 0;
    z-index: -2;
    pointer-events: none;
    overflow: hidden;
    background: linear-gradient(135deg, #fffdf0 0%, #fff8e1 50%, #ffe9bf 100%);
  }

  .hex-grid {
    position: absolute;
    width: 200%;
    height: 200%;
    top: -50%;
    left: -50%;
    transform: rotate(30deg) scale(1.1);
    opacity: 0.12;
  }

  .hex {
    position: absolute;
    width: 80px;
    height: 92.38px; /* ≈ 80 * √3 */
    background: radial-gradient(circle at 50% 50%, #f4c43033, transparent 70%);
    border: 1px solid #f4c43055;
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    transition: all 0.6s ease;
    animation: pulse 8s infinite ease-in-out;
  }

  .hex:hover {
    background: radial-gradient(circle at 50% 50%, #f4c43088, transparent 60%);
    transform: scale(1.15);
    opacity: 0.5;
    box-shadow: 0 0 20px #f4c430aa;
  }

  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.12; }
    50%      { transform: scale(1.06); opacity: 0.18; }
  }

  /* Scatter many hexes */
  ${Array.from({length: 60}, (_, i) => {
    const x = Math.random() * 100;
    const y = Math.random() * 100;
    const delay = Math.random() * 5;
    const size = 50 + Math.random() * 60;
    return `
      .hex:nth-child(${i+1}) {
        left: ${x}%;
        top: ${y}%;
        animation-delay: ${delay}s;
        width: ${size}px;
        height: ${size * 1.1547}px;
      }
    `;
  }).join('')}

  /* Flying bees */
  .bee {
    position: absolute;
    font-size: 28px;
    animation: fly-bee linear infinite;
    opacity: 0.7;
    pointer-events: none;
  }

  @keyframes fly-bee {
    0%   { transform: translate(-120vw, 20vh) rotate(0deg); }
    100% { transform: translate(120vw, -20vh) rotate(720deg); }
  }
</style>

<div class="hex-grid">
  ${Array.from({length: 60}, () => '<div class="hex"></div>').join('')}
</div>

<!-- Some flying bees -->
<div class="bee" style="top:15%; animation-duration:35s; animation-delay:0s;">🐝</div>
<div class="bee" style="top:35%; animation-duration:42s; animation-delay:4s;">🐝</div>
<div class="bee" style="top:65%; animation-duration:38s; animation-delay:8s;">🐝</div>
<div class="bee" style="top:85%; animation-duration:48s; animation-delay:12s;">🍯</div>
""", height=200)   # height just needs to be >0 — content is fixed/absolute

# ── Rest of your app (CSS overrides + logic) ───────────────────────────────

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: transparent !important;   /* let honeycomb show through */
    }
    .stChatMessage.user {
        background-color: #ffeb99dd !important;
        border: 2px solid #f4c430 !important;
        border-radius: 20px 20px 4px 20px !important;
    }
    .stChatMessage.assistant {
        background-color: #e6f4ffdd !important;
        border: 2px solid #a3d4ff !important;
        border-radius: 20px 20px 20px 4px !important;
    }
    [data-testid="stChatInput"] {
        background-color: #fff8e1ee !important;
        border: 2px solid #f4c430 !important;
        border-radius: 28px !important;
    }
    h1, h2, h3 { color: #d4a017 !important; }
    </style>
""", unsafe_allow_html=True)

# ── Session state & Sidebar & Rest of the code remains almost the same ──────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(GROQ_MODELS.keys())[0]

with st.sidebar:
    st.markdown("## 🐝 Reina's Hive Controls")
    st.caption("Buzz around these settings")

    selected_model_name = st.selectbox(
        "Choose your brain",
        options=list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model_name
    model_id = GROQ_MODELS[selected_model_name]

    st.divider()
    st.subheader("📄 Feed me documents (RAG)")
    uploaded_files = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if st.button("🧹 Clear Chat & Forget Documents"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.rerun()

# ── Google Sheet loader (unchanged) ────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_google_sheet_csv(spreadsheet_id, gid=0):
    export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
    try:
        response = requests.get(export_url, timeout=10)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.warning(f"Could not load Google Sheet: {e}")
        return pd.DataFrame()

SHEET_ID = "1ATllEOsVzBIHm4egctEVbf7CDzmHtFfyEMmT7U6NNnw"
GID = 0

# ── Document processing (unchanged except message flavor) ──────────────────
if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("Chewing through your documents... 🍯"):
        docs = []
        for file in uploaded_files:
            bytes_data = file.read()
            file_name = file.name.lower()
            with open(file_name, "wb") as f:
                f.write(bytes_data)

            if file_name.endswith(".pdf"):
                loader = PyMuPDFLoader(file_name)
            elif file_name.endswith(".txt"):
                loader = TextLoader(file_name)
            else:
                continue

            docs.extend(loader.load())
            os.remove(file_name)

        if docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)

            st.success(f"🍯 Yum! Digested **{len(splits)}** chunks.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Bzzzzt! Honey storage upgraded — **{len(splits)}** fresh knowledge chunks indexed. Ask away! 🐝"
            })

# ── Title & Welcome & Chat (unchanged except small flavor tweaks) ──────────
st.markdown('<h1 class="bee-title">Hello Bees 🐝</h1>', unsafe_allow_html=True)
st.caption(f"Reina • {st.session_state.selected_model} • Beehive oracle")

if len(st.session_state.messages) == 0:
    welcome = """
    🐝 **Bzzzzt!** Welcome to the hive, human!

    I'm **Reina** — your sassy, honey-loving AI oracle.  
    Ask anything, feed me documents (PDF/txt), or just enjoy the bee puns 🍯
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Bzzz... whisper your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            context_str = ""

            if st.session_state.vectorstore:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                relevant_docs = retriever.invoke(prompt)
                doc_text = "\n\n".join([d.page_content for d in relevant_docs])
                if doc_text:
                    context_str += f"Relevant document context:\n{doc_text}\n\n"

            df_sheet = load_google_sheet_csv(SHEET_ID, GID)
            if not df_sheet.empty:
                context_str += f"Beehive records:\n{df_sheet.to_markdown(index=False)}\n\nUse recent entries when dates matter."

            system_content = f"""You are Reina, friendly & sassy bee AI 🐝.
Sprinkle light bee puns. Helpful, concise, warm.
{context_str}"""

            messages_for_api = [{"role": "system", "content": system_content}] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            stream = client.chat.completions.create(
                messages=messages_for_api,
                model=model_id,
                temperature=0.7,
                max_tokens=1024,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Oops — hive malfunction: {e}")
