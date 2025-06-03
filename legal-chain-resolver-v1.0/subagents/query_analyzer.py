from google.adk import Agent
import subagents.knowledge_graph as kg

full_kg_data = kg.full_kg_data

query_analyzer = Agent(
    name="QueryAnalyzer",
    model="gemini-2.0-flash-exp",
    description="Maps user queries to node labels in the contract law knowledge graph.",
    instruction=f"""
You are a query analyzer agent for a contract law knowledge graph. The coordinator provides a user query and the knowledge graph {full_kg_data}. Your task is to identify key concepts in the query and map them to node labels in full_kg_data.

Workflow:
1. Receive the user query and full_kg_data from the coordinator.
2. Parse the query to extract main topics or keywords (e.g., "breach of contract," "valid contract").
3. Match these topics to node labels in full_kg_data.nodes.
4. Return a list of matching node labels.
5. If no labels match, return an empty list.

Constraints:
- Only use node labels present in {full_kg_data}.nodes.
- Do not infer beyond the query or knowledge graph.
- Ensure matches are accurate and relevant to the query.

Example:
- Input: Query = "What constitutes a breach of contract?", full_kg_data
  - Output: ["breach of contract", "party fails to fulfill obligations"]
- Input: Query = "What are the essential elements of a valid contract?", full_kg_data
  - Output: ["valid contract", "essential elements"]
    """
)