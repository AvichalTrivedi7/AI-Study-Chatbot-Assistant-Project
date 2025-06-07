# === Imports & Global Models ===
import tkinter as tk
from tkinter import scrolledtext
import faiss, numpy as np, threading
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# Pre-load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Load chunks and FAISS index ===
with open("vector_store/chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n\n")

index = faiss.read_index("vector_store/study_index.faiss")

# === Load Mistral Model === (Perfect for my computer specs)
llm = Llama(
    model_path="models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=8192,                 # Good choice for longer prompts
    max_tokens=3072,            # Generation upto these many tokens
    n_threads=10,               # Use all 12 logical threads (CPU cores × 2)
    n_batch=512,                # Very important — reduces forward pass loops
    use_mmap=True,              # Fast memory-mapped model loading
    use_mlock=True,             # Locks into RAM (Windows requires admin privileges or WSL2)
    temperature=0.7,
    top_p=0.95,
    repeat_penalty=1.1
)

# === FAISS Search (reusing existing embeddings) ===
def embed_query(text):
    # Dummy function: Replace with the actual embedding method used during indexing
    # For example, using sentence-transformers or the same model as used during indexing
    return model.encode([text])[0].astype("float32")

def search_similar_chunks(query, k=3):
    query_vector = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

# === Ask local Mistral LLM ===
def ask_mistral(context, query):
    prompt = f"""You are a helpful AI Study Assistant. Use the below context to answer the question.

Context:
{context}

Question:
{query}

Answer:"""
    response = llm(prompt, max_tokens=1024, stop=["</s>"])
    return response["choices"][0]["text"].strip()

# === Callback for Ask Button ===
def run_llm(query, context):
    try:
        answer = ask_mistral(context, query)
        chat_box.insert(tk.END, f"AI: {answer}\n", "ai")
        chat_box.see(tk.END)
    except Exception as e:
        chat_box.insert(tk.END, f"Error: {e}\n", "ai")
    finally:
        ask_button.config(state=tk.NORMAL)
        entry.config(state=tk.NORMAL)

def ask_question():
    query = entry.get()
    if not query.strip():
        return
    
    # disable inputs until model returns
    ask_button.config(state=tk.DISABLED)
    entry.config(state=tk.DISABLED)

    chat_box.insert(tk.END, f"\nYou: {query}\n", "user")
    chat_box.see(tk.END)
    entry.delete(0, tk.END)
    root.update_idletasks()

    try:
        context = "\n".join(search_similar_chunks(query))
        chat_box.insert(tk.END, f"AI: Thinking...\n", "ai")
        chat_box.see(tk.END)
        threading.Thread(target=run_llm, args=(query, context)).start()
    except Exception as e:
        chat_box.insert(tk.END, f"Error: {e}\n", "ai")

# === GUI Setup ===
root = tk.Tk()
root.title("AI Study Assistant")
root.geometry("500x700")

chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12))
chat_box.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
chat_box.tag_config("user", foreground="blue", font=("Helvetica", 11, "bold"))
chat_box.tag_config("ai", foreground="green", font=("Helvetica", 11, "bold"))

entry = tk.Entry(root, font=("Helvetica", 12))
entry.pack(padx=10, pady=5, fill=tk.X)
entry.bind("<Return>", lambda event: ask_question())

ask_button = tk.Button(root, text="Ask", command=ask_question, font=("Helvetica", 12, "bold"))
ask_button.pack(pady=5)

root.mainloop()
