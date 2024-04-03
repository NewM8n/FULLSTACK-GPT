import time
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)
st.title("DocumentGPT")

st.markdown("""
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
""")

file = st.file_uploader("Upload a .txt .pdf or .docs file", type=["pdf","txt","docx"])

if file:
    st.write(file)
    file_content = file.read()
    file_path = "./.cache/files/{file.name}"
    st.write(file_content, file_path)
    with open(file_path,"wb") as f:
        f.write(file_content)