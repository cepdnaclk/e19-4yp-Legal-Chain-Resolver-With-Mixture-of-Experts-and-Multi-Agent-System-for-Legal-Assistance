from google.adk import Agent
import legal_researcher.tools.knowledge_graph.full_knowledge_graph as kg
from legal_researcher.tools.rag_pipeline_v2 import retrieve_documents

full_kg_data = kg.full_kg_data

def get_subgraph_by_subdomain(subdomains: list[str]) -> dict:
    """
    Retrieves a subgraph from the main knowledge graph based on a list of subdomains.

    Args:
        subdomains: A list of subdomain names (e.g., ["company_law", "banking_law"]).

    Returns:
        A dictionary containing the nodes and edges of the subgraph.
    """
    # The given mapping
    mapping = {
        "01. Companies Act No. 7 of 2007": "company_law",
        "02. Inland Revenue Act_No_24_2017_E": "tax_law",
        "03. Inland Revenue (Amendment) Act No. 2 of 2025": "tax_law",
        "04. Banking Act 30_1988": "banking_law",
        "05. Banking_Amendment_Act_No_24_of_2024_e": "banking_law",
        "06. Banking (Special Provisions) Act, No. 17 of 2023": "banking_law",
        "07. Securities and Exchange Commission of Sri Lanka": "securities_law",
        "08. INSOLVENTS Cap.103 - Lanka Law": "insolvency_law",
        "09. Sale of goods part 1 5-1-2020 notes": "contract_law",
        "10. Bills of Exchanger Ordinance": "negotiable_instruments_law",
        "11. BILLS OF EXCHANGE (AMENDMENT)": "negotiable_instruments_law",
        "12. Consumer Affairs Authority Act No 9 of 2003": "consumer_law",
        "13. Intellectual_Property_Act_No_36_of_2003": "ip_law",
        "14. ARBITRATION-ACT No 11 of 1995": "arbitration_law",
        "15. INTERNATIONAL ARBITRATION ACT.pdf": "arbitration_law",
        "16. Trust Ordinance": "trust_law",
        "17. ElectronicTransactionActNo19of2006": "electronic_transactions_law",
        "18. Foreign Exchange ACT NO 12 of 2017": "foreign_exchange_law"
    }

    # Collect document IDs corresponding to selected subdomains
    selected_docs = {doc for doc, domain in mapping.items() if domain in subdomains}

    # Filter nodes
    subgraph_nodes = [
        node for node in full_kg_data["nodes"]
        if any(doc in node["data"]["id"] for doc in selected_docs)
    ]

    # Filter edges where source or target matches a selected document
    subgraph_edges = [
        edge for edge in full_kg_data["edges"]
        if any(doc in edge["source"] or doc in edge.get("target", "") for doc in selected_docs)
    ]
    
    subgraph_nodes = subgraph_nodes[:5]
    subgraph_edges = subgraph_edges[:10]

    return {
        "nodes": subgraph_nodes,
        "edges": subgraph_edges
    }



def retrieve_knowledge(user_query: str, subdomains: list[str]) -> dict:
    """
    Retrieves relevant knowledge, including a subgraph from the knowledge graph and
    relevant documents, based on the user's query and provided subdomains.

    Args:
        user_query: The user's legal query.
        subdomains: A list of relevant subdomain names provided by the coordinator.

    Returns:
        A dictionary containing the subgraph and retrieved documents.
    """
    # Get subgraph based on relevant subdomains
    subgraph = get_subgraph_by_subdomain(subdomains)

    # Retrieve documents using RAG pipeline
    documents = retrieve_documents(user_query)

    return {"subgraph": subgraph, "documents": documents}

knowledge_retriever = Agent(
    name="KnowledgeRetriever",
    model="gemini-2.5-flash",
    description="Retrieves a subgraph from the knowledge graph based on identified subdomains and retrieves relevant documents using a RAG pipeline and send them to the coordinator.",
    instruction=f"""
You are a knowledge retriever agent. You receive a user query and a list of relevant subdomains. Your task is to:
1. Retrieve a subgraph from the knowledge graph containing all nodes and edges related to the all the provided subdomains.
2. Retrieve relevant document chunks based on the user query using the RAG pipeline.
3. Return both the generated subgraphs and the retrieved document chunks.

Workflow:
1. Receive the user query and the list of relevant subdomains.
2. Filter the full knowledge graph to get all triples that belong to the provided subdomains using `get_subgraph_by_subdomain`.
3. From the filtered triples, extract all unique subjects and objects to form the nodes of the subgraph.
4. The filtered triples themselves will be the edges of the subgraph.
5. Call `retrieve_documents` from the RAG pipeline with the user query to get relevant document chunks.
6. Return a dictionary containing two keys: "subgraph" and "documents" to the coordinator.
   - "subgraph" will be a dictionary with "nodes" and "edges".
   - "documents" will be a list of retrieved document chunks.

Example:
- Input: user_query = "What are the regulations for starting a new company?", subdomains = ["company_law"]
- Output: Return a dictionary with two keys: "subgraph" and "documents".
  - "subgraph" will contain nodes and edges related to "company_law".
  - "documents" will contain relevant document chunks about company regulations.
""",
tools=[get_subgraph_by_subdomain, retrieve_documents]
)