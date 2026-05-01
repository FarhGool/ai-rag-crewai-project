# 🧠 AI RAG + CrewAI Project

## 📌 Project Overview
This project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline combined with CrewAI agents and Tavily web search.

The system can:
- Load and process documents (PDF)
- Convert text into embeddings
- Store and retrieve knowledge using FAISS
- Answer questions using RetrievalQA
- Fetch real-time AI trends from the web
- Generate structured reports using AI agents

---

## ⚙️ Technologies Used
- LangChain – Framework for LLM applications  
- FAISS – Vector database for similarity search  
- HuggingFace Embeddings – Convert text into vectors  
- Tavily API – Real-time web search  
- CrewAI – Multi-agent AI system  
- Groq LLM – Fast inference for text generation  
- Python  

---

## 🧩 Project Workflow
1. Load PDF document from `crew_data/doc.pdf`  
2. Split text into chunks using RecursiveCharacterTextSplitter  
3. Convert text chunks into embeddings using HuggingFace model  
4. Store embeddings in FAISS vector database  
5. Perform similarity search to retrieve relevant information  
6. Use RetrievalQA for question answering  
7. Use Tavily API to fetch latest AI trends  
8. Use CrewAI agent to generate structured report  

---

## ✅ Tasks Completed

### Task 1: Explore Vectorstore
- Used similarity_search() to retrieve relevant documents  
- Visualized embeddings using PCA and t-SNE  

### Task 2: Expand Knowledge Base
- Split documents into smaller chunks  
- Rebuilt FAISS vectorstore  
- Tested retrieval with multiple queries  

### Task 3: Create AI Agent
- Created Trend Analyst / Writer agent  
- Used Tavily API to get real-time AI trends  

### Task 4: Report Generator
- Generated structured markdown report using CrewAI  
- Output includes bullet points and summary  

---

## 📊 Example Output
# AI Trends Report (2026)

## Key Trends
- Rise of Multi-Agent Systems  
- Agentic AI & Smarter Automation  
- AI as a Collaborative Partner  

## Summary
AI is becoming more autonomous and collaborative, improving efficiency and decision-making across industries.

---

## 🚀 How to Run the Project

1. Install required libraries:
pip install -r requirements.txt

2. Create a `.env` file and add your API keys:
TAVILY_API_KEY=your_api_key  
GROQ_API_KEY=your_api_key  

3. Open and run:
analysis.ipynb

---

## 📁 Project Structure
ai-rag-project/
│
├── analysis.ipynb  
├── README.md  
├── .gitignore  
│  
└── crew_data/  
    └── doc.pdf  

---

## 🔒 Security Note
The `.env` file is not uploaded to GitHub to keep API keys secure.

---

## 🎯 Conclusion
This project demonstrates a real-world AI system combining document retrieval, real-time web search, and AI agents to generate meaningful insights.

It can be extended into chatbots, research assistants, or AI-powered applications.