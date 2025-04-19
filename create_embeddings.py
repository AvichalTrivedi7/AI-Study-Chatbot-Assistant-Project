import os
import faiss
import openai
import numpy as np
from dotenv import load_dotenv
from read_pdf import extract_text_from_pdf

# Load OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Split text into chunks
def chunk_text(text, max_tokens=500):
    sentences = text.split('. ')
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_tokens:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

# Get OpenAI embedding for a chunk
def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Main process
if __name__ == "__main__":
    pdf_path = "your_pdf_notes.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    print(f"📚 Total chunks: {len(chunks)}")

    embeddings = [get_embedding(chunk) for chunk in chunks]
    embeddings_np = np.array(embeddings).astype("float32")

    # Save using FAISS
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings_np)

    faiss.write_index(index, "vector_store/study_index.faiss")
    with open("vector_store/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")

    print("✅ Done! Vectors saved.")
