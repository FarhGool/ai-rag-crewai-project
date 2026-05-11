# =========================================================
# AGENTIC RAG APPLICATION
# Streamlit + LangChain + CrewAI + Tavily
# =========================================================

# =========================
# IMPORTS
# =========================

import os
import streamlit as st
from dotenv import load_dotenv

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain_classic.chains import RetrievalQA

# Tavily
from tavily import TavilyClient

# CrewAI
from crewai import Agent, Task, Crew, LLM

# PDF Export
from fpdf import FPDF


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Agentic RAG App",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Application")
st.write("Upload documents and ask questions using AI agents.")


# =========================================================
# CREATE UPLOAD FOLDER
# =========================================================

if not os.path.exists("uploads"):
    os.makedirs("uploads")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

search_option = st.sidebar.radio(
    "Choose Search Mode",
    [
        "Document Search",
        "Web Search",
        "Both"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


# =========================================================
# LOAD FREE LLM
# =========================================================

@st.cache_resource
def load_llm():

    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


llm = load_llm()



crew_llm = LLM(
    model="ollama/llama3", 
    base_url="http://localhost:11434"
)


# =========================================================
# LOAD DOCUMENTS
# =========================================================

vectorstore = None
qa_chain = None

if uploaded_file is not None:

    # Save uploaded file
    save_path = os.path.join("uploads", uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    # Load PDF
    loader = PyPDFLoader(save_path)

    docs = loader.load()


    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = text_splitter.split_documents(docs)

    chunks = [
        c for c in chunks
        if len(c.page_content.strip()) > 50
    ]


    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Vectorstore
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )


    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )


# =========================================================
# TAVILY WEB SEARCH
# =========================================================

tavily = TavilyClient(api_key=TAVILY_API_KEY)


# =========================================================
# USER QUESTION
# =========================================================

question = st.text_input(
    "Ask a question"
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if st.button("Generate Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        document_answer = ""
        web_answer = ""

        # =================================================
        # DOCUMENT SEARCH
        # =================================================

        if search_option in ["Document Search", "Both"]:

            if qa_chain is None:

                st.warning("Please upload a PDF first.")

            else:

                with st.spinner("Searching documents..."):

                    document_answer = qa_chain.run(question)

        # =================================================
        # WEB SEARCH
        # =================================================

        if search_option in ["Web Search", "Both"]:

            with st.spinner("Searching internet..."):

                response = tavily.search(query=question)

                results = response["results"][:5]

                web_answer = ""

                for r in results:

                    web_answer += f"""
Title: {r['title']}

Content:
{r['content']}

Source:
{r['url']}

--------------------------------------------------
"""

        # =================================================
        # CREATE AGENTS
        # =================================================

        research_agent = Agent(

            role="Research Agent",

            goal="""
            Analyze and explain information
            from documents and internet sources.
            """,

            backstory="""
            You are an expert AI researcher
            skilled at understanding documents,
            extracting insights,
            and explaining information clearly.
            """,

            llm=crew_llm,

            verbose=False
        )

        web_agent = Agent(

            role="Web Research Agent",

            goal="""
            Analyze internet search results
            and identify important insights,
            trends,
            technologies,
            and useful information.
            """,

            backstory="""
            You are an internet research expert.
            You specialize in analyzing
            web search results,
            AI trends,
            emerging technologies,
            and real-time information.
            """,

            llm=crew_llm,

            verbose=False
        )

        writer_agent = Agent(

            role="Writer Agent",

            goal="""
            Write detailed, professional,
            and easy-to-understand answers.
            """,

            backstory="""
            You are a professional technical writer.
            You create detailed AI reports
            and clear explanations.
            """,

            llm=crew_llm,

            verbose=False
        )

        critic_agent = Agent(

            role="Critic Agent",

            goal="""
            Improve the final answer
            by making it clearer,
            more detailed,
            and more professional.
            """,

            backstory="""
            You review AI-generated content
            and improve quality,
            readability,
            and completeness.
            """,

            llm=crew_llm,

            verbose=False
        )

        # =================================================
        # RESEARCH TASK
        # =================================================

        research_task = Task(

            description=f"""
            User Question:
            {question}

            DOCUMENT RESULTS:
            {document_answer}

            WEB RESULTS:
            {web_answer}

            Analyze all available information.

            Extract:
            - important insights
            - explanations
            - key findings
            - useful examples

            Provide detailed research findings.
            """,

            expected_output="""
            A detailed research summary.
            """,

            agent=research_agent
        )

        # =================================================
        # WEB TASK
        # =================================================

        web_task = Task(

            description=f"""
            Analyze these internet search results:

            {web_answer}

            Identify:
            - important insights
            - AI trends
            - technologies
            - useful examples
            - future developments

            Explain them clearly.
            """,

            expected_output="""
            A detailed web research summary.
            """,

            agent=web_agent
        )

        # =================================================
        # WRITER TASK
        # =================================================

        writer_task = Task(

            description=f"""
            Create a professional detailed answer
            for the following user question:

            QUESTION:
            {question}

            Use:
            - document research
            - internet research

            The answer must:
            - be detailed
            - be professional
            - explain concepts clearly
            - include sections
            - include bullet points when useful
            - provide examples when possible

            Use markdown formatting.
            """,

            expected_output="""
            A detailed professional response.
            """,

            agent=writer_agent
        )

        # =================================================
        # CRITIC TASK
        # =================================================

        critic_task = Task(

            description="""
            Improve the final response.

            IMPORTANT RULES:
            - ONLY return the final improved answer
            - DO NOT explain what you changed
            - DO NOT say:
              "Here is the rewritten version"
              "Here is the polished response"
              or similar sentences
            - DO NOT include reasoning
            - DO NOT include introductions

            Make the response:
            - professional
            - detailed
            - clean
            - easy to understand
            - well formatted

            Remove repetition and improve readability.

            Return ONLY the final polished answer.
            """,

            expected_output="""
            A clean professional final response only.
            """,

            agent=critic_agent
        )

        # =================================================
        # CREATE CREW
        # =================================================

        crew = Crew(

            agents=[
                research_agent,
                web_agent,
                writer_agent,
                critic_agent
            ],

            tasks=[
                research_task,
                web_task,
                writer_task,
                critic_task
            ],

            verbose=False
        )

        # =================================================
        # RUN CREW
        # =================================================

        with st.spinner("AI Agents are working..."):

            result = crew.kickoff()

        # =================================================
        # DISPLAY RESULT
        # =================================================

        #st.markdown("## 📌 Final Answer")

        st.markdown(result.raw)

        # =================================================
        # EXPORT TO PDF
        # =================================================

        pdf = FPDF()

        pdf.add_page()

        pdf.set_font("Arial", size=12)

        lines = result.raw.split("\n")

        for line in lines:
            pdf.multi_cell(0, 10, line)

        pdf_path = "final_report.pdf"

        pdf.output(pdf_path)

        # =================================================
        # DOWNLOAD BUTTON
        # =================================================

        with open(pdf_path, "rb") as file:

            st.download_button(
                label="📥 Download PDF Report",
                data=file,
                file_name="final_report.pdf",
                mime="application/pdf"
            )