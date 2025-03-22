import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# --- SETUP API ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# --- SETUP CHROMADB ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="knowledge_base")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- TEXT UTILITIES ---
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def add_folder_pdfs_to_chroma(folder_path="docs"):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            chunks = split_text(text)
            embeddings = embedding_model.encode(chunks).tolist()
            ids = [f"{filename}_{i}" for i in range(len(chunks))]
            collection.add(documents=chunks, embeddings=embeddings, ids=ids)
            print(f"[INFO] Processed: {filename}")

# --- RETRIEVAL ---
def retrieve_relevant_docs(query, top_k=3):
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0]

# --- ASK GEMINI ---
def ask_gemini(query):
    context_docs = retrieve_relevant_docs(query)
    context_text = "\n".join(context_docs)
    prompt = f"""Gunakan informasi berikut untuk menjawab pertanyaan:

{context_text}

Pertanyaan: {query}
Jawaban:"""
    response = model.generate_content(prompt)
    return response.text
