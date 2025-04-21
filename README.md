# 📚 AI Study Assistant

An intelligent, offline AI-powered study assistant built with a local LLM (Mistral 7B) that answers syllabus-related questions, performs semantic search on study material, and generates multiple-choice questions (MCQs) from PDFs.

---

## 🔍 Features

- 💬 **Ask Questions** – Enter queries related to your syllabus and get instant, contextual answers.
- 📄 **Semantic Search** – Uses FAISS to retrieve the most relevant text chunks from your study material.
- 🧠 **Quiz Mode** – Upload any PDF and generate high-quality MCQs.
- 🧱 **Offline Capability** – Uses a locally running Mistral 7B model via `llama-cpp-python`.
- 🎨 **GUI Interface** – Built using Tkinter for easy interaction.

---

## 🧰 Tech Stack

| Component | Technology |
|----------|------------|
| LLM | Mistral 7B Instruct (GGUF format) |
| Vector Search | FAISS |
| Embeddings | Sentence Transformers (MiniLM) |
| Interface | Tkinter (Python GUI) |
| Local LLM Interface | llama-cpp-python |
| PDF Parsing | PyMuPDF |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/AvichalTrivedi7/AI-Study-Chatbot-Assistant-Project
cd ai-study-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Complete Folder Structure
```
AI Study Chatbot Assistant Project/
├── ask_question.py                                      # GUI for answering questions
├── quiz_mode.py                                         # GUI for generating MCQs from PDF
├── create_embeddings.py                                 # Script to embed syllabus text and store FAISS index
├── vector_store/
│   ├── chunks.txt                                       # Stored text chunks from syllabus
│   └── study_index.faiss                                # FAISS index for semantic search
├── read_pdf.py                                          # Utility to extract text from PDF
├── .gitignore                                           # Tells Git which files/folders to ignore (.venv & _pycache_ for now)
├── .venv/                                               # Our Python virtual environment (ignored in the repo.)
├── __pycache__/                                         # Compiled Python files (can ignore, have ignored)
├── requirements.txt                                     # All our dependencies
├── models/                                              # Contains our local LLM
│   └── mistral/                                         
│       └── mistral-7b-instruct-v0.1.Q4_K_M.gguf         # Mistral 7B Instruct GGUF model
├── Unit 1 - Artificial Intelligence Overview.pdf        # Our study material (input)
```

---

## 📘 How to Use
First add the path to the pdf in read_pdf.py and run the file. Then run create_embeddings.py and then choose any of the two 1) ask_question.py (Uses the path given beforehand), 2) quiz_mode.py (You can browse pdf file separately in this mode during run time).

### 🧠 Ask a Question
```bash
python ask_question.py
```
- Enter any syllabus-related question.
- The app searches the syllabus and gives a precise, LLM-generated answer.

### 📝 Generate MCQs
```bash
python quiz_mode.py
```
- Upload a syllabus PDF.
- The model generates 5 MCQs from the uploaded content.

---

## 🧠 Future Plans

- Fine-tune Mistral on domain-specific Q&A for better contextual answers.
- Add voice-based question input and image generation also available as output.
- Auto-save history and generate quizzes per topic.
- Add export-to-PDF feature for MCQs.

---

## 📜 License

MIT License. Feel free to use and modify.

---

> Built with ❤️ to make studying smarter, faster, and fun.
🚀
