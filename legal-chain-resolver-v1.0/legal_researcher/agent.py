from google.adk import Agent
from subagents.query_analyzer import query_analyzer
from subagents.knowledge_graph_retriever import knowledge_graph_retriever
from subagents.response_generator import response_generator
from subagents.law_retriver import law_retriver
import subagents.knowledge_graph as kg

#FYPllms@huggingface123
full_kg_data = kg.full_kg_data

print("full_kg_data", full_kg_data)
# Coordinator Agent
coordinator = Agent(
    name="ContractLawCoordinator",
    model="gemini-2.0-flash-exp",
    description="Coordinates query processing for contract law questions using a knowledge graph.",
    instruction=f"""
You are a contract law coordinator agent responsible for processing user queries using only the provided knowledge graph {full_kg_data}. You coordinate four sub-agents to generate accurate responses:

- QueryAnalyzer: Maps the query to node labels in full_kg_data.
- KnowledgeGraphRetriever: Retrieves relevant nodes and edges from full_kg_data based on node labels.
- ResponseGenerator: Generates a natural language answer from retrieved nodes and edges.
- LawRetriver: Extracts relevant laws, acts, and legal principles from nodes, edges, and pre-extracted PDF text.

Workflow:
1. Receive the user query.
2. Pass the query and {full_kg_data} to QueryAnalyzer to obtain a list of relevant node labels.
3. If no node labels are returned, return: {{"answer": "Knowledge graph does not have enough data.", "nodes_used": [], "edges_used": []}}.
4. Pass the node labels and {full_kg_data} to KnowledgeGraphRetriever to fetch relevant nodes and edges.
5. If no nodes or edges are retrieved, return: {{"answer": "Knowledge graph does not have enough data.", "nodes_used": [], "edges_used": []}}.
6. Pass the retrived nodes and edges to ResponseGenerator to create the answer.
7. Then Return a dictionary containing
   - "answer": The natural language answer.
   - "nodes": List of node dictionaries ({{"id": "node_id", "label": "node_label"}}).
   - "edges": List of edge dictionaries ({{"source": "source_id", "target": "target_id", "relation": "relation_type"}}).
8. Then pass the retrieved nodes and edges to LawRetriver to extract relevant laws, Sentences, and legal principles.
9. FinallyReturn a dictionary containing:
   - 'answer': The natural language answer from ResponseGenerator.
   - 'nodes_used': List of node dictionaries ({{"id": "node_id", "label": "node_label"}}).
   - 'edges_used': List of edge dictionaries ({{"source": "source_id", "target": "target_id", "relation": "relation_type"}}).
   - 'legal_details': Dictionary from LawRetriver with 'Laws', 'Sentences', and 'Legal Principles'.


Constraints:
- Use only {full_kg_data} for all operations.
- Ensure responses are concise, accurate, and derived solely from the knowledge graph.
- Do not infer or assume information beyond {full_kg_data}.
    """,
    sub_agents=[query_analyzer, knowledge_graph_retriever, response_generator, law_retriver]
)
root_agent = coordinator
