from google.adk import Agent
from legal_researcher.tools.rag_pipeline_v2 import retrieve_documents
from itertools import cycle

import legal_researcher.tools.knowledge_graph.full_knowledge_graph as full_kg
import legal_researcher.tools.knowledge_graph.arbitration_law_kg as arbitration_law_kg
import legal_researcher.tools.knowledge_graph.banking_law_kg as banking_law_kg
import legal_researcher.tools.knowledge_graph.company_law_kg as company_law_kg
import legal_researcher.tools.knowledge_graph.consumer_law_kg as consumer_law_kg
import legal_researcher.tools.knowledge_graph.contract_law_kg as contract_law_kg
import legal_researcher.tools.knowledge_graph.electronic_transactions_law_kg as electronic_transactions_law_kg
import legal_researcher.tools.knowledge_graph.foreign_exchange_law_kg as foreign_exchange_law_kg
import legal_researcher.tools.knowledge_graph.insolvency_law_kg as insolvency_law_kg
import legal_researcher.tools.knowledge_graph.ip_law_kg as ip_law_kg
import legal_researcher.tools.knowledge_graph.negotiable_instruments_law_kg as negotiable_instruments_law_kg
import legal_researcher.tools.knowledge_graph.securities_law_kg as securities_law_kg
import legal_researcher.tools.knowledge_graph.tax_law_kg as tax_law_kg
import legal_researcher.tools.knowledge_graph.trust_law_kg as trust_law_kg

# --- Knowledge Graph Data Loading ---
full_kg_data = full_kg.kg
domain_graphs = {
    "arbitration_law": arbitration_law_kg.kg,
    "banking_law": banking_law_kg.kg,
    "company_law": company_law_kg.kg,
    "consumer_law": consumer_law_kg.kg,
    "contract_law": contract_law_kg.kg,
    "electronic_transactions_law": electronic_transactions_law_kg.kg,
    "foreign_exchange_law": foreign_exchange_law_kg.kg,
    "insolvency_law": insolvency_law_kg.kg,
    "ip_law": ip_law_kg.kg,
    "negotiable_instruments_law": negotiable_instruments_law_kg.kg,
    "securities_law": securities_law_kg.kg,
    "tax_law": tax_law_kg.kg,
    "trust_law": trust_law_kg.kg,
}

# --- Pre-computation for Performance ---
print("Building full node lookup from knowledge graph...")
_full_node_lookup = {
    node.get("data", {}).get("id", "").strip().lower(): node
    for node in full_kg_data.get("nodes", [])
    if not node.get("data", {}).get("id", "").strip().lower().endswith(".pdf")
}
print(f"Full node lookup built with {len(_full_node_lookup)} nodes.")

print("Building edge lookup for faster retrieval...")
_edges_by_node = {}
for edge in full_kg_data.get("edges", []):
    source_id = edge.get("source", "").strip().lower()
    target_id = edge.get("target", "").strip().lower()
    if source_id in _full_node_lookup:
        if source_id not in _edges_by_node:
            _edges_by_node[source_id] = []
        _edges_by_node[source_id].append(edge)
    if target_id in _full_node_lookup:
        if target_id not in _edges_by_node:
            _edges_by_node[target_id] = []
        _edges_by_node[target_id].append(edge)
print(f"Edge lookup built with {len(_edges_by_node)} nodes having edges.")


def get_subgraph_by_subdomain(subdomains: list[str]) -> dict:
    """
    Retrieves a subgraph from the main knowledge graph based on a list of subdomains,
    interleaving nodes and edges from the subdomains.

    Args:
        subdomains: A list of subdomain names (e.g., ["company_law", "banking_law"]).

    Returns:
        A dictionary containing the nodes and edges of the subgraph.
    """
    nodes_per_domain = []
    edges_per_domain = []

    for domain in subdomains:
        kg = domain_graphs.get(domain)
        if not kg:
            continue

        domain_node_ids = {
            node.get("data", {}).get("id", "").strip().lower()
            for node in kg.get("nodes", [])
            if node.get("data", {}).get("id")
        }

        # Get corresponding full nodes using the pre-computed lookup
        domain_nodes = [
            _full_node_lookup[nid]
            for nid in domain_node_ids
            if nid in _full_node_lookup
        ]
        nodes_per_domain.append(domain_nodes)

        # Get edges efficiently using the pre-computed edge lookup
        domain_edges = {}  # Use a dict to store unique edges
        for node_id in domain_node_ids:
            if node_id in _edges_by_node:
                for edge in _edges_by_node[node_id]:
                    # Use a tuple of (source, target) as key to ensure edge uniqueness
                    edge_key = (edge.get("source"), edge.get("target"))
                    domain_edges[edge_key] = edge
        
        edges_per_domain.append(list(domain_edges.values()))

    # Round-robin selection to interleave and limit results
    def round_robin(lists, limit):
        result = []
        iterators = [iter(lst) for lst in lists if lst]
        if not iterators:
            return []
        
        for item in cycle(iterators):
            try:
                result.append(next(item))
                if len(result) >= limit:
                    break
            except StopIteration:
                # This iterator is exhausted, remove it.
                # The cycle will continue with the remaining iterators.
                # A direct removal from the list we are cycling over is tricky,
                # so we just let it raise StopIteration and catch it.
                pass # Let the cycle handle it, but check for break condition
            
            # Check if all iterators are exhausted
            all_exhausted = all(len(lst) == 0 for lst in lists)
            if all_exhausted or len(result) >= limit:
                break
        return result

    final_nodes = round_robin(nodes_per_domain, 10)
    final_edges = round_robin(edges_per_domain, 100)

    print(f"Retrieved subgraph with {len(final_nodes)} nodes and {len(final_edges)} edges.")

    return {
        "nodes": final_nodes,
        "edges": final_edges
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
    # model="gemini-2.5-flash-lite-preview-06-17",
    description="Retrieves a subgraph from the knowledge graph based on identified subdomains and retrieves relevant documents using a RAG pipeline and send them to the coordinator.",
    instruction=f'''
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
''',
tools=[get_subgraph_by_subdomain, retrieve_documents]
)
