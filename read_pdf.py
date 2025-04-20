import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text.replace("\n", " ") + " "  # replace newlines with spaces
    return ' '.join(text.split())  # normalize spaces

# Test this file alone by running:
if __name__ == "__main__":
    pdf_path = "C:/Users/HP/Documents/College 1st Year (Repositories)/AI Study Chatbot Assistant Project/Unit 1 - Artificial Intelligence Overview.pdf"
    text = extract_text_from_pdf(pdf_path)
    print(text)  # Print all characters
