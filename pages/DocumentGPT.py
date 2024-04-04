import time
import os
from uuid import UUID
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""
    message_box = st.empty()

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args,**kwargs):
        self.message += token
        self.message_box.markdown(self.message)



@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    os.makedirs(f"./.cache/files/", exist_ok=True)
    file_path = f"./.cache/files/{file.name}"
    with open(file_path,"wb") as f:
        f.write(file_content)
    cache_dir=LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["openai_key"])

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message,role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
def save_api_key(api_key):
    st.session_state["openai_key"] = openai_key

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

        Context: {context}
        """,
    ),
    ("human","{question}")
])

st.title("DocumentGPT")

st.markdown("""
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
            
Upload your files on the sidebar.
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docs file", type=["pdf","txt","docx"])
    openai_key = st.text_input("Insert your OPENAI_API_KEY...")
    button = st.button("KEY 저장")
    github_url = st.text("https://github.com/NewM8n/FULLSTACK-GPT")
    app_url = st.text("https://fullstack-gpt-newm8n.streamlit.app/DocumentGPT")

    if button:
        save_api_key(openai_key)
        st.write(f"API_KEY = {openai_key}")
        if openai_key == "":
            st.warning("OPEN_API_KEY 를 넣어주세요.")

if openai_key:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        openai_api_key=st.session_state["openai_key"]
    )
else:
    st.markdown("PLEASE WRITE OPENAI_API_KEY")

if file and openai_key :
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!","ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = {
            "context" : retriever | RunnableLambda(format_docs),
            "question" : RunnablePassthrough()
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []