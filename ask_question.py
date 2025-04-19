# ask_question.py

import pickle
import faiss
from llama_cpp import Llama

# Load FAISS index
faiss_index = faiss.read_index("vector_store/study_index.faiss")


# Load chunks
with open("vector_store/chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.readlines()


# Load local Mistral model (adjust path if needed)
llm = Llama(
    model_path="C:/Users/HP/Documents/College 1st Year (Repositories)/AI Study Chatbot Assistant Project/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  
    n_ctx=2048,
    n_threads=6,  # adjust based on your CPU
    n_gpu_layers=20,  # optional, if you have a GPU
    verbose=False
)

def search_query(query, k=3):
    # Load same embedding model as during indexing
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query])
    D, I = faiss_index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

def ask(question):
    top_chunks = search_query(question)
    context = "\n\n".join(top_chunks)

    prompt = f"""[INST] You are an AI Study Assistant. Use the given context to answer the question clearly and concisely.

Context:
{context}

Question: {question}
Answer: [/INST]"""

    response = llm(prompt, max_tokens=512, temperature=0.7)
    print("\n📘 Answer:\n", response["choices"][0]["text"].strip())

# === Test ===
if __name__ == "__main__":
    while True:
        user_q = input("\n❓ Ask your question (or type 'exit'): ")
        if user_q.lower() == "exit":
            break
        ask(user_q)
