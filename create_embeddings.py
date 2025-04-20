import os
import faiss
import numpy as np
from read_pdf import extract_text_from_pdf
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Split text into chunks
def chunk_text(text, max_chunk_length=500):
    sentences = text.split('. ')
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_chunk_length:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

# Get embedding from Hugging Face model
def get_embedding(text):
    return model.encode(text)

# Main process
if __name__ == "__main__":
    pdf_path = "C:/Users/HP/Documents/College 1st Year (Repositories)/AI Study Chatbot Assistant Project/Unit 1 - Artificial Intelligence Overview.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    print(f"📚 Total chunks: {len(chunks)}")

    embeddings = [get_embedding(chunk) for chunk in chunks]
    embeddings_np = np.array(embeddings).astype("float32")

    # Save using FAISS
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, "vector_store/study_index.faiss")

    with open("vector_store/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")

    print("✅ Done! Vectors saved.")
