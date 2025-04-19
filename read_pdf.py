import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Test this file alone by running:
if __name__ == "__main__":
    pdf_path = "your_pdf_notes.pdf"  # replace with your actual PDF name
    text = extract_text_from_pdf(pdf_path)
    print(text[:1000])  # Print first 1000 characters
