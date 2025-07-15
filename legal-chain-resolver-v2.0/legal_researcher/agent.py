from google.adk import Agent
from legal_researcher.subagents.query_analyzer import query_analyzer #, knowledge_graph_retriever, response_generator, law_retriver

# Coordinator Agent
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.5-flash",
    description="Coordinates query processing for contract law questions.",
    instruction=f"""
You are a contract law coordinator agent. You are responsible for processing user queries by coordinating a team of sub-agents.

Workflow:
1. Receive the user query.
2. Pass the query to the QueryAnalyzer to get a list of relevant subdomains.
3. Pass the list of subdomains to the KnowledgeGraphRetriever to get a relevant subgraph.
4. Pass the subgraph and the original query to the ResponseGenerator to get a natural language answer.
5. Pass the subgraph to the LawRetriever to get a list of relevant laws and acts.
6. Return a dictionary containing the answer and the list of laws and acts.
""",
sub_agents=[query_analyzer]  # Add other sub-agents as needed
)

def process_query(query: str) -> dict:
    """
    Processes a user query and returns a response.

    Args:
        query: The user's query.

    Returns:
        A dictionary containing the answer and a list of relevant laws and acts.
    """
    subdomains = query_analyzer.get_relevant_subdomains(query)
    # subgraph = knowledge_graph_retriever.get_subgraph_by_subdomain(subdomains)
    # answer = response_generator.generate_response(subgraph, query)
    # laws_and_acts = law_retriver.get_laws_and_acts(subgraph)

    return {
        "answer": subdomains
        # "laws_and_acts": laws_and_acts
    }

root_agent = coordinator
