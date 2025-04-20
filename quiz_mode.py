import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from read_pdf import extract_text_from_pdf
from llama_cpp import Llama

# Load Mistral model
mistral_model_path = "models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
llm = Llama(
    model_path="models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=8192,   # You can try 6144 or 8192
    max_tokens=3072,  # Set higher if desired, within n_ctx limit
    temperature=0.7,
    top_p=0.95,
    repeat_penalty=1.1
)

# GUI App
class QuizGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quiz Mode - Generate MCQs from PDF")

        self.pdf_path = ""

        # Buttons
        self.upload_btn = tk.Button(root, text="📂 Browse PDF", command=self.browse_pdf, width=20)
        self.upload_btn.pack(pady=10)

        self.generate_btn = tk.Button(root, text="🧠 Generate MCQs", command=self.generate_mcqs, width=20)
        self.generate_btn.pack(pady=10)

        # Output Area
        self.output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=30)
        self.output_text.pack(padx=10, pady=10)

    def browse_pdf(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf")],
            title="Select a Syllabus PDF"
        )
        if file_path:
            self.pdf_path = file_path
            messagebox.showinfo("PDF Selected", f"✅ PDF selected:\n{file_path}")

    def generate_mcqs(self):
        if not self.pdf_path:
            messagebox.showerror("No PDF Selected", "Please upload a PDF file first.")
            return

        try:
            text = extract_text_from_pdf(self.pdf_path)
            text = text[:1500] # Trimmed for speed  

            prompt = f"""
You are an intelligent tutor. Based on the following text, generate a 5 multiple-choice questions.
 Each question should have 4 options (A, B, C, D). Clearly give the correct answers below each question.

Text:
{text}

Return only the questions, options and answers in a clean, readable format.
"""

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "🔄 Generating MCQs, please wait...\n\n")
            self.root.update()

            output = llm(prompt=prompt, max_tokens=3072, stop=["</s>"])
            mcqs = output["choices"][0]["text"].strip()

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, mcqs)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = QuizGeneratorApp(root)
    root.mainloop()
