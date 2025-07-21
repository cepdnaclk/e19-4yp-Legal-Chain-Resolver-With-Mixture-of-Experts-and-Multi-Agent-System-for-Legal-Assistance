# Legal Chain Resolver With Mixture of Experts and Multi-Agent System for Legal Assistance

#### Team

- E/19/310, Ranage R.D.P.R., [email](mailto:e19310@eng.pdn.ac.lk)
- E/19/426, Weerasinghe P.M., [email](mailto:e19426@eng.pdn.ac.lk)
- E/19/304, Pushpakumara R.M.S.P., [email](mailto:e19304@eng.pdn.ac.lk)

#### Supervisors

- Dr. Damayanthi Herath, [email](mailto:damayanthiherath@eng.pdn.ac.lk)
- Ms. Yasodha Vimukthi, [email](mailto:yashodhav@eng.pdn.ac.lk)

## Abstract

The Sri Lankan legal system faces significant barriers in accessibility, affordability, and efficiency, particularly within commercial law. High legal costs, language barriers, and limited digital legal resources make timely and accurate legal assistance challenging for many, especially SMEs. Existing AI-driven legal solutions in Sri Lanka predominantly rely on Natural Language Processing (NLP), which lacks structured legal reasoning and is prone to hallucinations. This project proposes an AI-driven legal assistance framework focused on Sri Lankan commercial law, integrating a Mixture-of-Experts (MoE) model, a multi-agent system, and knowledge graphs. The system is designed to enhance legal accuracy, interpretability, and efficiency by dynamically routing legal queries to domain-specific AI experts, structuring legal knowledge, and simulating real-world legal workflows. This approach aims to provide accessible, reliable, and context-aware legal support for businesses, legal professionals, and policymakers in Sri Lanka, ultimately improving legal decision-making and compliance.

## 1. Project Overview

This project aims to create an AI-powered legal assistant that specializes in Sri Lankan commercial law. The system is designed to be accurate, interpretable, and efficient. It uses a combination of advanced AI techniques to achieve this:

*   **Mixture-of-Experts (MoE) Model:** This model is used to route legal queries to the most appropriate AI expert. Each expert is specialized in a specific area of commercial law, ensuring that the user receives the most accurate and relevant information.
*   **Multi-Agent System:** This system simulates the workflow of a legal team. It consists of several AI agents, each with a specific role, such as query analysis, knowledge retrieval, and response generation. This allows the system to handle complex legal queries in a structured and efficient manner.
*   **Knowledge Graphs:** These are used to represent legal knowledge in a structured way. This allows the system to reason about legal concepts and relationships, leading to more accurate and interpretable responses.
*   **Retrieval-Augmented Generation (RAG) Pipeline:** This pipeline is used to retrieve relevant legal documents and information from a knowledge base. This ensures that the system's responses are always up-to-date and based on the latest legal information.

## 2. Getting Started

### 2.1. Prerequisites

*   Python 3.10 or higher
*   Pip (Python package installer)
*   Git

### 2.2. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### 2.3. Running the Application

The main application is a Streamlit-based user interface. To run it, use the following command:

```bash
streamlit run implementations/user-interface/legal_researcher_app.py
```

This will start a web server and open the application in your browser.

## 3. Project Structure

The project is organized into the following directories:

```
├───.gitignore
├───README.md
├───requirements.txt
├───.venv/
├───code/
├───data/
│   └───law_documents/
├───docs/
├───implementations/
│   ├───knowledge-graph-manual-v1.0/
│   ├───knwoledge-graph-neo4j-v1.0/
│   ├───moe-model/
│   ├───rag-pipeline/
│   └───user-interface/
├───legal-chain-resolver-v1.0/
├───legal-chain-resolver-v2.0/
├───legal-chain-resolver-v3.0/
├───legal-chain-resolver-v3.1/
└───results/
```

*   **`.gitignore`:** A file that specifies which files and directories to ignore in a Git repository.
*   **`README.md`:** This file.
*   **`requirements.txt`:** A file that lists the Python libraries required to run the project.
*   **`.venv/`:** A directory that contains a virtual Python environment for the project.
*   **`code/`:** A directory that contains the source code for the project.
*   **`data/`:** A directory that contains the data used in the project, such as legal documents.
*   **`docs/`:** A directory that contains the documentation for the project.
*   **`implementations/`:** A directory that contains the implementations of the core components of the project, such as the knowledge graph, MoE model, RAG pipeline, and user interface.
*   **`legal-chain-resolver-v*`:** A series of directories that contain the different versions of the multi-agent system.
*   **`results/`:** A directory that contains the results of the project, such as comparisons of the different versions of the system.

## 4. Key Components

### 4.1. Mixture-of-Experts (MoE) Model

The MoE model is a key component of the system. It is responsible for routing legal queries to the most appropriate AI expert. Each expert is specialized in a specific area of commercial law, such as contract law, company law, or intellectual property law. This ensures that the user receives the most accurate and relevant information.

The MoE model is implemented in the `implementations/moe-model/` directory. The `README.md` file in that directory provides a detailed explanation of the model's architecture and how to train and use it.

### 4.2. Multi-Agent System

The multi-agent system simulates the workflow of a legal team. It consists of several AI agents, each with a specific role:

*   **Query Analyzer:** This agent is responsible for analyzing the user's query and identifying the key legal issues.
*   **Knowledge Retriever:** This agent is responsible for retrieving relevant legal information from the knowledge base.
*   **Law Retriever:** This agent is responsible for retrieving relevant legal documents from the knowledge base.
*   **Response Generator:** This agent is responsible for generating a response to the user's query.

The multi-agent system is implemented in the `legal-chain-resolver-v3.1/` directory. The `agent.py` file in that directory defines the main coordinator agent and the sequence of sub-agents.

### 4.3. Knowledge Graph

The knowledge graph is used to represent legal knowledge in a structured way. It consists of a set of nodes and edges, where the nodes represent legal concepts and the edges represent the relationships between them. This allows the system to reason about legal concepts and relationships, leading to more accurate and interpretable responses.

The knowledge graph is implemented using Neo4j. The `implementations/knwoledge-graph-neo4j-v1.0/` directory contains the initial implementation of the knowledge graph. The `legal-chain-resolver-v3.1/legal_researcher/tools/knowledge_graph/` directory contains the latest version of the knowledge graph, with separate Python scripts for different legal domains.

### 4.4. Retrieval-Augmented Generation (RAG) Pipeline

The RAG pipeline is used to retrieve relevant legal documents and information from a knowledge base. This ensures that the system's responses are always up-to-date and based on the latest legal information.

The RAG pipeline has evolved from a basic implementation in `implementations/rag-pipeline/rag-pipeline.py` to a more advanced version in `legal-chain-resolver-v3.1/legal_researcher/tools/rag_pipeline_v2.1.py`. The latest version uses a FAISS vector store for efficient similarity search and automatically rebuilds the index if the underlying PDF documents are updated.

## 5. Evolution of the Project

The project has evolved through several versions:

*   **v1.0:** The initial version of the project.
*   **v2.0:** This version introduced the MoE model and the multi-agent system.
*   **v3.0:** This version improved the knowledge graph and the RAG pipeline.
*   **v3.1:** The current version of the project, which includes several improvements and bug fixes.

## 6. Dependencies

The project requires the following Python libraries:

*   arxiv==2.2.0
*   beautifulsoup4==4.13.4
*   bs4==0.0.2
*   cassio==0.1.10
*   chromadb==1.0.15
*   faiss-cpu==1.11.0
*   fastapi==0.116.1
*   flask==3.1.1
*   fpdf==1.7.2
*   google-adk==1.6.1
*   google-generativeai==0.8.5
*   groq==0.30.0
*   jupyter==1.1.1
*   langchain-google-genai==2.0.10
*   langchain-groq==0.3.6
*   langchain_community==0.3.27
*   langchain_core==0.3.68
*   langchain_openai==0.3.27
*   langchainhub==0.1.21
*   langchain-huggingface==0.3.0
*   langserve==0.3.1
*   networkx==3.5
*   numpy==2.3.1
*   pandas==2.3.1
*   pdfplumber==0.11.7
*   pypdf==5.8.0
*   pypdf2==3.0.1
*   python-dotenv==1.1.1
*   scikit-learn==1.7.0
*   sentence-transformers==5.0.0
*   spacy==3.8.7
*   sse_starlette==2.4.1
*   streamlit==1.46.1
*   torch==2.7.1
*   uvicorn==0.35.0
*   wikipedia==1.4.0
