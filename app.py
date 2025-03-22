# app.py
import streamlit as st
from rag_engine import ask_gemini, add_folder_pdfs_to_chroma

st.set_page_config(page_title="Knowledge Management RAG", layout="wide")
st.title("📚 Knowledge Management Assistant")

with st.spinner("📦 Memuat semua dokumen PDF..."):
    add_folder_pdfs_to_chroma("docs")
    st.success("✅ Semua dokumen berhasil dimuat ke knowledge base!")

query = st.text_input("Tanyakan sesuatu berdasarkan isi dokumen:")

if query:
    with st.spinner("🧠 Memikirkan jawaban..."):
        answer = ask_gemini(query)
        st.markdown("### 💬 Jawaban:")
        st.write(answer)
