import tkinter as tk
from tkinter import filedialog, messagebox
import re
import numpy as np
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====================================================
# PDF PROCESSING FUNCTIONS
# ====================================================
def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def load_pdf(path):
    try:
        reader = PyPDF2.PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return clean_text(text)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load PDF\n{e}")
        return ""

# ====================================================
# MAIN APPLICATION
# ====================================================
class PDFQnAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Question Answering")
        self.root.geometry("700x400")
        self.root.configure(bg="#1e1e1e")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.pdf_text = ""
        self.chunks = []
        self.index = None

        # Load PDF Button
        self.load_btn = tk.Button(
            root, text="📂 Load PDF", font=("Segoe UI", 14),
            bg="#00bfff", fg="white", command=self.load_pdf_gui
        )
        self.load_btn.pack(pady=10)

        # Question Entry
        self.question_entry = tk.Entry(
            root, font=("Segoe UI", 14), width=60,
            bg="#222", fg="white", insertbackground="white"
        )
        self.question_entry.pack(pady=10)
        self.question_entry.bind("<Return>", lambda event: self.answer_question())

        # Answer Display
        self.answer_label = tk.Label(
            root, text="Answer will appear here",
            font=("Segoe UI", 12), bg="#1e1e1e", fg="white",
            wraplength=650, justify="left"
        )
        self.answer_label.pack(pady=20)

        # Ask Button
        self.ask_btn = tk.Button(
            root, text="Ask", font=("Segoe UI", 12),
            bg="#0066cc", fg="white", command=self.answer_question
        )
        self.ask_btn.pack(pady=5)

    # ====================================================
    # Load PDF
    # ====================================================
    def load_pdf_gui(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file_path:
            return

        self.pdf_text = load_pdf(file_path)
        if not self.pdf_text:
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=350, chunk_overlap=80, length_function=len
        )
        self.chunks = splitter.split_text(self.pdf_text)

        embeddings = self.model.encode(
            self.chunks, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        dim = embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, 50)
        self.index.train(embeddings)
        self.index.add(embeddings)

        messagebox.showinfo("Success", f"PDF loaded! Total chunks: {len(self.chunks)}")

    # ====================================================
    # Answer Question
    # ====================================================
    def answer_question(self):
        query = self.question_entry.get().strip()
        if not query:
            return
        if not self.index or not self.chunks:
            messagebox.showwarning("Warning", "Please load a PDF first")
            return

        # Embed query
        q_embed = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        distances, ids = self.index.search(q_embed, 5)

        # Re-ranking with cosine similarity
        results = []
        for idx in ids[0]:
            chunk = self.chunks[idx]
            score = util.cos_sim(
                self.model.encode(query, convert_to_tensor=True),
                self.model.encode(chunk, convert_to_tensor=True)
            ).item()
            results.append((score, chunk))

        results = sorted(results, key=lambda x: x[0], reverse=True)
        context_text = " ".join([text for score, text in results[:3]])

        # Pick most relevant sentence
        sentences = context_text.split(". ")
        if sentences:
            best_sentence = max(sentences, key=lambda s: util.cos_sim(
                self.model.encode(query, convert_to_tensor=True),
                self.model.encode(s, convert_to_tensor=True)
            ).item())
            answer_text = best_sentence.strip() + "."
        else:
            answer_text = "Sorry, no answer found in the PDF."

        self.answer_label.config(text=answer_text)


# ====================================================
# RUN APPLICATION
# ====================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = PDFQnAApp(root)
    root.mainloop()
