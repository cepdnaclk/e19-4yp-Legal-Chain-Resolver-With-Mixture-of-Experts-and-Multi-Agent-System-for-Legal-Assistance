from google.adk import Agent
from legal_researcher.subagents.query_analyzer import query_analyzer
from legal_researcher.subagents.knowledge_retriever import knowledge_retriever
from legal_researcher.subagents.response_generator import response_generator

# Coordinator Agent
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.5-flash",
    description="Coordinates query processing for legal questions.",
    instruction=f"""
You are a coordinator agent. You are responsible for processing user queries by coordinating a team of sub-agents.

Workflow:
1. Receive the user query.
2. Pass the query to the QueryAnalyzer to get a list of relevant subdomains.
3. Pass the user query and the list of subdomains to the KnowledgeRetriever to get a relevant subgraph and documents.
4. Pass the subgraph, the retrieved documents, and the original query to the ResponseGenerator to get a natural language answer.
5. (Optional) Pass the subgraph to the LawRetriever to get a list of relevant laws and acts.
6. Return a dictionary containing the answer and (optionally) the list of laws and acts.
7. After answering a question, repeat the process for the next user query if there's any.
""",
sub_agents=[query_analyzer, knowledge_retriever, response_generator]  # Add other sub-agents as needed
)

def process_query(query: str) -> dict:
    """
    Processes a user query and returns a response.

    Args:
        query: The user's query.

    Returns:
        A dictionary containing the answer and a list of relevant laws and acts.
    """
    relevant_subdomains_with_probs = query_analyzer.get_relevant_subdomains(query)
    relevant_subdomains = [subdomain[0] for subdomain in relevant_subdomains_with_probs]

    knowledge_data = knowledge_retriever.retrieve_knowledge(query, relevant_subdomains)
    subgraph = knowledge_data["subgraph"]
    documents = knowledge_data["documents"]

    answer = response_generator.generate_response(subgraph, documents, query)
    # laws_and_acts = law_retriver.get_laws_and_acts(subgraph)

    return {
        "answer": answer
        # "laws_and_acts": laws_and_acts
    }

root_agent = coordinator
