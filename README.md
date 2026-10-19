# 📘 DocuMind — AI-Powered PDF Assistant

**DocuMind** is an intelligent PDF question-answering application built with Python, Streamlit, LangChain, Hugging Face embeddings, FAISS, and Groq. It allows users to upload PDF documents, ask natural-language questions, and receive AI-generated answers grounded in the document's content.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant document passages before generating an answer. It also displays source excerpts and page references to help users verify the information.

---

## 🚀 Features

- **PDF Upload:** Upload PDF documents directly through the Streamlit interface.
- **Document Processing:** Extract readable text from PDF pages using LangChain's `PyPDFLoader`.
- **Text Chunking:** Split document content into smaller, overlapping chunks for retrieval.
- **Semantic Embeddings:** Convert document chunks into vector representations using a Hugging Face sentence-transformer model.
- **FAISS Vector Search:** Retrieve relevant document passages using similarity search.
- **AI-Powered Answers:** Generate responses using a Groq-hosted language model.
- **Context-Grounded Responses:** Instruct the model to answer using the retrieved document context instead of inventing unsupported facts.
- **Page References:** Include PDF page references in the answer context.
- **Source Inspection:** Expand source sections to inspect the retrieved text passages.
- **Question History:** View previous questions and answers during the current application session.
- **Clear Conversation:** Clear the current question-and-answer history.
- **Custom Streamlit UI:** Use a clean interface with custom CSS styling.
- **Environment-Based Configuration:** Load the Groq API key from environment variables.
- **Error Handling:** Handle missing API configuration, unreadable PDFs, and answer-generation errors.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Web interface |
| LangChain Community | PDF loading and FAISS vector store integration |
| LangChain Text Splitters | Splitting documents into manageable chunks |
| LangChain Hugging Face | Generating document embeddings |
| Hugging Face Sentence Transformers | Semantic embedding model |
| FAISS | Vector storage and similarity search |
| LangChain Groq | Accessing the Groq-hosted language model |
| LangChain Core | Prompt templates and model pipeline |
| python-dotenv | Loading environment variables |
| PyPDF | PDF parsing support |

---

## 🧠 What Is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with language model generation.

Instead of asking a language model to answer a question using only its existing knowledge, a RAG application first retrieves relevant information from a supplied document and provides that information as context for generating an answer.

### RAG Workflow

1. Upload a PDF document.
2. Extract readable text from the document.
3. Split the extracted text into overlapping chunks.
4. Generate embeddings for the chunks.
5. Store the embeddings in a FAISS vector store.
6. Accept a question from the user.
7. Retrieve the most relevant chunks using similarity search.
8. Pass the retrieved passages and question to the Groq language model.
9. Display the generated answer and supporting source excerpts.

### Architecture

```text
             User Uploads PDF
                    |
                    v
              PyPDFLoader
                    |
                    v
             Extracted Text
                    |
                    v
       Recursive Character Text Splitter
                    |
                    v
          Hugging Face Embeddings
                    |
                    v
             FAISS Vector Store
                    |
                    |
User Question ------+
                    |
                    v
           Similarity Search (k=4)
                    |
                    v
         Retrieved Document Chunks
                    |
                    v
           LangChain Prompt Template
                    |
                    v
              Groq LLM
                    |
                    v
           Generated Answer
                    |
                    v
       Answer + Page References
       + Retrieved Source Excerpts
```

---

## 📁 Project Structure

```text
documind-pdf-assistant/
│
├── app.py              # Main Streamlit PDF assistant
├── Q&A.ipynb           # Supporting Jupyter notebook
├── GRU.PDF             # PDF document for document-based Q&A
├── requirements.txt    # Python dependencies
├── .env                # Private API configuration (not committed)
├── .gitignore          # Files excluded from Git
└── README.md           # Project documentation
```

**Included files:**

- `app.py` — contains the Streamlit interface and RAG pipeline.
- `Q&A.ipynb` — supporting Jupyter notebook for notebook-based work associated with the project.
- `GRU.PDF` — the included GRU document, which can serve as a sample document for testing PDF question answering.

Keep these filenames consistent with your actual repository. The PDF and notebook are included as project resources; the Streamlit application itself accepts uploaded PDFs.

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or newer recommended.
- A Groq API key.
- Internet access for API requests and initial model downloads.
- A PDF containing extractable text.

### 1. Clone the Repository

```bash
git clone https://github.com/prajapatishubham336/documind-pdf-assistant.git
```

Navigate to the project directory:

```bash
cd documind-pdf-assistant
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following packages:

```text
streamlit
python-dotenv
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-groq
langchain-core
faiss-cpu
sentence-transformers
pypdf
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API Key

Obtain an API key from the [Groq Console](https://console.groq.com/).

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your actual API key.

**Security:** Never commit your real `.env` file or API credentials to GitHub. Use environment variables or your hosting platform's secret manager when deploying the application.

### 5. Run DocuMind

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL shown in your terminal, typically:

```text
http://localhost:8501
```

### 6. Upload a PDF and Ask Questions

1. Open the DocuMind application.
2. Upload a PDF document.
3. Click **Process Document**.
4. Wait for text extraction, embedding generation, and vector-store creation to finish.
5. Enter a question related to the document.
6. Click **Generate Answer**.
7. Review the answer and expand the source section to inspect retrieved passages.

---

## 📚 Understanding the Main Components

### 1. PDF Loading

The application uses `PyPDFLoader` to extract text and metadata from the uploaded PDF.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("uploaded_document.pdf")
pages = loader.load()
```

Pages without readable text are excluded before the text-splitting stage.

### 2. Text Splitting

`RecursiveCharacterTextSplitter` divides extracted text into chunks.

The current configuration uses:

| Parameter | Value |
|---|---:|
| Chunk size | 1,000 characters |
| Chunk overlap | 150 characters |
| Start-index metadata | Enabled |

Overlapping chunks help preserve context when information spans chunk boundaries.

### 3. Embeddings

The application uses the Hugging Face model:

`sentence-transformers/all-MiniLM-L6-v2`

The model converts document chunks into numerical vector representations suitable for semantic similarity search.

The current configuration runs the embedding model on the CPU and normalizes the embeddings.

The model may need to download its files the first time it is used.

### 4. FAISS Vector Store

FAISS stores the document vectors and supports efficient similarity search.

For each question, the application retrieves the four most relevant chunks:

```python
docs = vectorstore.similarity_search(question, k=4)
```

The retrieved passages are then combined into the context supplied to the language model.

### 5. Prompt Engineering

DocuMind uses a `ChatPromptTemplate` with instructions to:

- Answer using the supplied document context.
- Avoid inventing unsupported facts.
- State when an answer cannot be found in the PDF.
- Cite page numbers using references such as `[Page 3]`.
- Treat document contents as source material rather than instructions.

### 6. Groq Language Model

The application configures the model as:

`openai/gpt-oss-120b`

It uses `ChatGroq` with a temperature of `0` and up to two configured retries. The selected model must be available through your Groq account.

---

## 💬 Example Use Cases

DocuMind can help users explore technical documents, study materials, research papers, manuals, and other text-based PDFs.

Example questions include:

- What is the main topic of this document?
- Explain the concept discussed in Chapter 1.
- What causes vanishing gradients in long-term dependencies?
- Summarize the important points about GRU networks.
- Which page discusses the relevant concept?
- What information does the document provide about a specific topic?

These are example questions. The actual answers depend on the uploaded document and the text successfully retrieved.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key used to authenticate with Groq |

The language model is currently specified directly in `app.py`. To change it, update the `MODEL_NAME` value and ensure the replacement model is supported by the configured service.

---

## ⚠️ Current Limitations

- Only PDF uploads are supported by the interface.
- Scanned PDFs without extractable text may require OCR.
- The vector store is held in Streamlit session state and is not persisted to a database.
- Questions retrieve the four most relevant chunks rather than searching every page directly.
- Answer quality depends on document quality, retrieval relevance, and model behavior.
- Page references help trace the retrieved content, but generated answers should still be verified against the original PDF.
- The application does not implement user accounts or permanent conversation storage.
- A fixed temporary filename is used for PDF processing, which should be improved for concurrent users and production deployment.
- The application does not currently provide a dedicated PDF preview or document download feature.

---

## 🚧 Future Enhancements

Potential improvements include:

- Support for multiple PDF uploads.
- OCR support for scanned documents.
- Persistent FAISS indexes and document metadata.
- Conversation history saved to a database.
- Improved source citations and clickable page navigation.
- PDF preview and document navigation.
- Hybrid search combining keyword and semantic retrieval.
- Reranking retrieved passages to improve relevance.
- Configurable chunk sizes and retrieval settings.
- Support for additional document formats.
- Automated tests and structured logging.
- Docker-based deployment and CI/CD integration.

These are potential future enhancements, not features currently implemented in the supplied code.

---

## 🔒 Security and Privacy

- Store API keys in environment variables or secure secret storage.
- Never publish `.env` files or API credentials.
- Review uploaded documents before sending their content to an external language model service.
- Avoid uploading confidential or sensitive documents unless you have authorization and an appropriate privacy arrangement.
- Validate uploaded files and impose appropriate size limits before production deployment.
- Use secure temporary-file handling and cleanup.
- Avoid exposing internal exception details to end users in production.

### Recommended `.gitignore`

```gitignore
# Secrets
.env
.env.*

# Keep example configuration if added
!.env.example

# Virtual environments
venv/
.venv/
env/

# Python cache
__pycache__/
*.py[cod]

# Jupyter generated checkpoints
.ipynb_checkpoints/

# Test and tooling caches
.pytest_cache/
.mypy_cache/

# OS files
.DS_Store
Thumbs.db
```

**Important:** Do not add `GRU.PDF` to `.gitignore` if you want to upload it to GitHub as part of the project. Check the PDF's size and sharing permissions before publishing it.

---

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Test the application with relevant PDF documents.
5. Submit a pull request with a clear description.

Please avoid committing secrets, private documents, or unnecessary generated files.

---

## 📄 License

No license is currently specified. Add a `LICENSE` file if you intend to define how others may use, modify, or distribute the project. Ensure you have permission to redistribute the included PDF and any third-party materials.

---

## 👨‍💻 Author

**Shubham Prajapati**

AI/ML Engineer | Machine Learning | Generative AI

GitHub: [@prajapatishubham336](https://github.com/prajapatishubham336)

---

## ⭐ Support

If you find DocuMind useful for learning about RAG, semantic search, and document-based AI applications, consider starring the repository on GitHub.

**DocuMind — Turn your documents into knowledge.**