# DocuMind AI — PDF Question Answering with RAG

A Streamlit application that answers questions about a PDF using LangChain, HuggingFace sentence-transformer embeddings, FAISS vector search and Groq inference.

## Requirements

- Python 3.10 or 3.11 recommended
- A Groq API key: https://console.groq.com/keys
- Internet access for Groq API calls and the first download of the embedding model

## Setup on Windows

Open Command Prompt or PowerShell in this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and replace `your_groq_api_key_here` with your real API key. Keep the key private and do not upload `.env` to GitHub.

Run:

```powershell
streamlit run app.py
```

Streamlit will show a local URL, normally `http://localhost:8501`.

## How to use

1. Enter the Groq API key in the sidebar (or put it in `.env`).
2. Upload a text-based PDF.
3. Click **Build / replace knowledge base** and wait for indexing to finish.
4. Ask questions in the chat input.
5. Expand **Sources used** to see the retrieved page excerpts.

## Suggested test PDF

Use the free book *An Introduction to Statistical Learning: with Applications in Python* from its official website:

https://www.statlearning.com/

Try asking:
- What is supervised learning?
- Explain the difference between classification and regression.
- What is cross-validation?
- Which page passages support the answer?

## Project structure

```text
documind_rag/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## RAG architecture

PDF -> PyPDF text extraction -> LangChain text splitting -> Sentence Transformers embeddings -> FAISS similarity search -> relevant passages -> Groq chat model -> answer with page references.

## Notes and limitations

- The PDF must contain selectable text. Scanned/image-only PDFs need OCR first.
- Embeddings run locally on CPU; the first run downloads the model and can take a few minutes.
- Groq API use may be subject to account limits and model availability. If the default model is unavailable, choose the alternative in the sidebar.
- This starter app builds an in-memory FAISS index each time you index a PDF. The index is not persisted between app restarts.
- Page references are based on the PDF page order (starting at page 1), not necessarily the printed page number in the book.
- This is a starter project, not a hardened multi-user production service. Do not upload confidential documents unless you are permitted to send their relevant excerpts to the selected LLM provider.
