# 🤖 Agentic RAG Application using LangChain, CrewAI, Streamlit & Ollama

---

# 📌 Project Overview

This project demonstrates the implementation of a complete **Agentic Retrieval-Augmented Generation (RAG) System** using:

- LangChain
- CrewAI
- FAISS Vector Database
- HuggingFace Embeddings
- Tavily Web Search
- Ollama Local LLM
- Streamlit Frontend

The application combines:
- document intelligence,
- semantic search,
- real-time web retrieval,
- and multi-agent AI collaboration

to create a powerful AI assistant capable of:
- answering user questions,
- analyzing uploaded documents,
- retrieving relevant information,
- generating detailed reports,
- and performing internet research.

The project follows the coursework requirements while also extending the system with additional real-world AI application features.

---

# 🎯 Objectives of the Project

The main goals of this project are:

✅ Build a Retrieval-Augmented Generation (RAG) pipeline  
✅ Process and search custom PDF documents  
✅ Create a semantic vector database using embeddings  
✅ Perform similarity-based document retrieval  
✅ Integrate real-time internet search using Tavily  
✅ Create multiple AI agents using CrewAI  
✅ Generate detailed AI-powered responses  
✅ Build an interactive frontend using Streamlit  
✅ Export generated reports into PDF format  

---

# 🧠 What is RAG (Retrieval-Augmented Generation)?

Retrieval-Augmented Generation (RAG) is an AI architecture that combines:

1. Information Retrieval
2. Large Language Models (LLMs)

Instead of relying only on the LLM’s training knowledge, RAG retrieves relevant information from external sources such as:
- PDFs
- databases
- websites
- vectorstores

before generating the final response.

This improves:
- accuracy,
- contextual understanding,
- factual consistency,
- and domain-specific performance.

---

# 🧩 Core Technologies Used

| Technology | Purpose |
|---|---|
| LangChain | Framework for building LLM applications |
| CrewAI | Multi-agent AI orchestration |
| FAISS | High-performance vector database |
| HuggingFace Embeddings | Convert text into vector embeddings |
| Streamlit | Interactive web application frontend |
| Tavily API | Real-time web search |
| Ollama | Local LLM inference |
| Transformers | HuggingFace model loading |
| Python | Backend programming language |
| FPDF | PDF report generation |

---

# ⚙️ System Architecture

The application workflow follows these steps:

## Step 1 — Document Upload
The user uploads PDF files through the Streamlit interface.

---

## Step 2 — Document Processing
The uploaded PDFs are:
- loaded,
- cleaned,
- and split into smaller text chunks.

This is done using:
```python
RecursiveCharacterTextSplitter
```

---

## Step 3 — Embedding Generation
Each chunk is converted into a vector embedding using:

```python
all-MiniLM-L6-v2
```

from HuggingFace sentence transformers.

---

## Step 4 — Vector Database Creation
The embeddings are stored inside a FAISS vector database.

This enables:
- semantic search,
- similarity retrieval,
- contextual document lookup.

---

## Step 5 — Retrieval-Augmented QA
When the user asks a question:
- the system retrieves the most relevant chunks,
- passes them to the LLM,
- and generates a contextual response.

---

## Step 6 — Web Search Integration
If enabled by the user:
- Tavily API searches the internet,
- retrieves real-time information,
- and provides current web context.

---

## Step 7 — Multi-Agent AI Workflow
CrewAI coordinates multiple AI agents.

Each agent performs a specialized task:
- research,
- writing,
- reviewing,
- refinement.

---

## Step 8 — Final Response Generation
The system generates:
- detailed,
- professional,
- structured,
- human-readable responses.

---

## Step 9 — PDF Report Export
The final response can be downloaded as a PDF report.

---

# 🤖 AI Agents Implemented

The project uses multiple collaborative AI agents.

---

# 1️⃣ Research Agent

## Role
AI Researcher

## Responsibilities
- Analyze retrieved document information
- Analyze internet search results
- Extract important insights
- Understand technical content
- Summarize findings

## Purpose
Acts as the primary information analysis agent.

---

# 2️⃣ Writer Agent

## Role
Professional Technical Writer

## Responsibilities
- Generate detailed answers
- Create structured reports
- Format professional responses
- Explain concepts clearly

## Purpose
Transforms research into polished readable content.

---

# 3️⃣ Critic Agent

## Role
Content Reviewer

## Responsibilities
- Improve clarity
- Improve readability
- Remove repetition
- Refine structure
- Ensure professionalism

## Purpose
Improves the quality of the final AI-generated response.

---

# 📂 Project Structure

```bash
RAG-Agent-Project/
│
├── analysis.ipynb
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── crew_data/
│   ├── doc.pdf
│   ├── AI Education.pdf
│   └── AI healthcare.pdf
│
├── uploads/
│
└── final_report.pdf
```

---

# 📄 Description of Files

| File | Description |
|---|---|
| analysis.ipynb | Main coursework notebook containing all tasks |
| app.py | Streamlit frontend application |
| requirements.txt | Python dependencies |
| README.md | Project documentation |
| .env | Stores API keys |
| .gitignore | Prevents sensitive files from being uploaded |
| crew_data/ | Knowledge base PDF files |
| uploads/ | User uploaded documents |

---

# 🔍 Coursework Tasks Completed

---

# ✅ Task 1 — Explore the Vectorstore

## Implemented Features
- Similarity search using FAISS
- Embedding analysis
- PCA visualization
- t-SNE visualization

## Example
```python
results = vectorstore.similarity_search(query, k=3)
```

## Outcome
Successfully demonstrated:
- semantic retrieval,
- embedding clustering,
- and vector analysis.

---

# ✅ Task 2 — Expand the Knowledge Base

## Implemented Features
- Added multiple PDF documents
- Rebuilt FAISS vectorstore
- Tested retrieval with multiple queries

## Added Documents
- AI Education.pdf
- AI healthcare.pdf
- doc.pdf

## Outcome
Improved:
- retrieval diversity,
- contextual coverage,
- and answer quality.

---

# ✅ Task 3 — Create New AI Agents

## Implemented Features
Created:
- Research Agent
- Writer Agent
- Critic Agent

## Technologies Used
- CrewAI
- Tavily API
- Ollama LLM

## Outcome
Enabled collaborative multi-agent workflows.

---

# ✅ Task 4 — Report Generator Agent

## Implemented Features
- Professional report generation
- Markdown formatting
- PDF export

## Outcome
Generated clean AI reports automatically.

---

# 🌐 Additional Features Implemented

The coursework was extended with additional real-world features:

## ✅ Streamlit Frontend
Interactive UI for:
- document upload,
- question answering,
- report generation.

---

## ✅ User Upload Feature
Users can upload their own PDF documents dynamically.

---

## ✅ Internet Search Option
Users can choose:
- Document Search
- Web Search
- Both

---

## ✅ Detailed AI Responses
Agents generate:
- longer,
- more detailed,
- more professional answers.

---

## ✅ PDF Export
Users can download generated reports.

---

# 📊 Embedding Visualization

The project visualizes embeddings using:

## PCA
Principal Component Analysis reduces embeddings into 2D space.

## t-SNE
t-SNE provides non-linear embedding visualization.

These visualizations help analyze:
- semantic relationships,
- clustering,
- document similarity.

---

# 🦙 Why Ollama?

Ollama is used to run LLMs locally.

Advantages:
- completely free,
- privacy-friendly,
- no API cost,
- offline capable.

This project uses:

```bash
phi3
```

as the local LLM model.

---

# 🌐 Why Tavily?

Tavily provides:
- real-time web search,
- current internet information,
- live AI trend retrieval.

This improves:
- freshness,
- relevance,
- dynamic knowledge access.

---

# 🚀 Installation Guide

---

# 1️⃣ Clone Repository

```bash
git clone <your-github-link>
```

---

# 2️⃣ Open Project Folder

```bash
cd RAG-Agent-Project
```

---

# 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# 4️⃣ Create .env File

Create a `.env` file and add:

```env
TAVILY_API_KEY=your_tavily_api_key
```

---

# 5️⃣ Install Ollama

Download:

https://ollama.com/download

---

# 6️⃣ Pull Phi3 Model

```bash
ollama pull phi3
```

---

# 7️⃣ Verify Ollama

```bash
ollama list
```

You should see:

```bash
phi3
```

---

# ▶️ Run Streamlit App

```bash
streamlit run app.py
```

---

# 🧪 Example Questions

```text
What is Machine Learning?

Explain AI in healthcare.

What are the latest AI trends in 2026?

How does generative AI work?

Summarize the uploaded document.

What are multi-agent AI systems?
```

---

# 📥 PDF Export

The application automatically generates downloadable PDF reports.

Generated using:
```python
FPDF
```

---

# 🔒 Security Notes

The following files are excluded using `.gitignore`:

- `.env`
- API keys
- virtual environments
- cache files

This prevents sensitive information from being uploaded to GitHub.

---

# 📄 requirements.txt

```text
streamlit
langchain
langchain-community
langchain-huggingface
langchain-text-splitters
langchain-classic
crewai
faiss-cpu
sentence-transformers
transformers
torch
accelerate
tavily-python
pypdf
python-dotenv
fpdf
scikit-learn
matplotlib
```

---

# 📚 Learning Outcomes

This project demonstrates practical knowledge of:

- Retrieval-Augmented Generation (RAG)
- Vector databases
- Semantic search
- Embeddings
- AI agents
- Multi-agent systems
- Web search integration
- Streamlit frontend development
- Local LLM deployment
- PDF generation
- AI workflow orchestration

---

# 🏁 Conclusion

This project successfully combines:
- AI agents,
- semantic retrieval,
- vector databases,
- real-time web search,
- and local LLMs

to create a powerful AI-powered research assistant.

The system demonstrates real-world applications of:
- RAG architectures,
- multi-agent collaboration,
- and intelligent document analysis.

The project can be further extended into:
- enterprise AI assistants,
- educational copilots,
- healthcare assistants,
- legal document analyzers,
- research assistants,
- and business intelligence systems.