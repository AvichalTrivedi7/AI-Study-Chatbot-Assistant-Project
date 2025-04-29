import tkinter as tk
from tkinter import scrolledtext
import faiss
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# === Load chunks and FAISS index ===
with open("vector_store/chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n\n")

index = faiss.read_index("vector_store/study_index.faiss")

# === Load LLM (Mistral) ===
llm = Llama(
    model_path="models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=8192,
    max_tokens=3072,
    temperature=0.7,
    top_p=0.95,
    repeat_penalty=1.1
)

# === Sentence Transformer Model for Embeddings ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(text):
    return embedding_model.encode([text])[0].astype("float32")

# === Search similar chunks from FAISS index ===
def search_similar_chunks(query, k=3):
    query_vector = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

# === Use LLM to simplify medical input based on context ===
def simplify_medical_text(user_input):
    similar_chunks = search_similar_chunks(user_input, k=3)
    context = "\n\n".join(similar_chunks)

    prompt = f"""You are a medical assistant that explains medical text in simple, patient-friendly, layman language.

Patient's Question:
{user_input}

Context from medical documents:
{context}

Now explain the input in layman terms. Keep it short, clear, and simple.
Layman:"""

    response = llm(prompt, max_tokens=3072, stop=["</s>"])
    return response["choices"][0]["text"].strip()

# === GUI Callback ===
def convert_text():
    medical_input = entry.get()
    if not medical_input.strip():
        return
    chat_box.insert(tk.END, f"\nMedical: {medical_input}\n", "user")
    chat_box.see(tk.END)
    entry.delete(0, tk.END)
    root.update_idletasks()

    try:
        simplified = simplify_medical_text(medical_input)
        chat_box.insert(tk.END, f"Layman: {simplified}\n", "ai")
        chat_box.see(tk.END)
    except Exception as e:
        chat_box.insert(tk.END, f"Error: {e}\n", "ai")

# === Build GUI ===
root = tk.Tk()
root.title("Medical Text Simplifier")
root.geometry("500x700")

chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12))
chat_box.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
chat_box.tag_config("user", foreground="purple", font=("Helvetica", 11, "bold"))
chat_box.tag_config("ai", foreground="dark green", font=("Helvetica", 11, "bold"))

entry = tk.Entry(root, font=("Helvetica", 12))
entry.pack(padx=10, pady=5, fill=tk.X)
entry.bind("<Return>", lambda event: convert_text())

ask_button = tk.Button(root, text="Simplify", command=convert_text, font=("Helvetica", 12, "bold"))
ask_button.pack(pady=5)

root.mainloop()
