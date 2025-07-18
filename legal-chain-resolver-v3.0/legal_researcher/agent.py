from google.adk.agents import SequentialAgent
from legal_researcher.subagents.query_analyzer import query_analyzer
from legal_researcher.subagents.knowledge_retriever import knowledge_retriever
from legal_researcher.subagents.response_generator import response_generator
from legal_researcher.subagents.law_retriever import law_retriever

# Coordinator Agent
coordinator = SequentialAgent(
    name="Coordinator",
    # model="gemini-2.5-flash",
    description="Coordinates query processing for legal questions.",
#     instruction=f"""
# You are a coordinator agent. You are responsible for processing user queries by coordinating a team of sub-agents.

# Workflow:
# DO NOT REPEAT YOURSELF, DO NOT GENERATE THE SAME OUTPUT FOR THE SAME QUERY, DO NOT REPEAT THE SAME TASK FOR THE SAME QUERY.
# FOLLOW THE PIPELINE STRICTLY WITHOUT SKIPPING ANY STEPS.

# 1. Receive the user query.
# 2. Pass the query to the `query_analyzer` agent to get a list of relevant subdomains.
# 3. Pass the user query and the list of subdomains to the `knowledge_retriever` agent to get a relevant subgraphs and documents.
# 4. Pass the subgraphs, documents, and query to the `law_retriever` agent to get a list of relevant laws, acts and relevant law sentences/clauses.
# 5. Pass the subgraphs, the retrieved documents, the original query, and the list of laws, acts and relevant law sentences/clauses to the `response_generator` agent to get a natural language answer.
# 6. `response_generator` agent will return a comprehensive answer that includes the referred acts and laws.
# 7. Save each user query and its corresponding answer in a log file called `query_log.json` for future reference.
# 8. If the user has more queries, repeat the process for the next query.

# Pipeline:
# User(query) -> Coordinator(agent)
# Coordinator(agent) -> QueryAnalyzer(agent)
# QueryAnalyzer(agent) -> get_top_subdomains(tool)
# get_top_subdomains(tool) -> QueryAnalyzer(agent)
# QueryAnalyzer(agent) -> Coordinator(agent)
# Coordinator(agent) -> KnowledgeRetriever(agent)
# KnowledgeRetriever(agent) -> [get_subgraph_by_subdomain(tool) + retrieve_documents(tool)]
# [get_subgraph_by_subdomain(tool) + retrieve_documents(tool)] -> KnowledgeRetriever(agent)
# KnowledgeRetriever(agent) -> Coordinator(agent)
# Coordinator(agent) -> LawRetriever(agent)
# LawRetriever(agent) -> get_laws_and_acts(tool)
# get_laws_and_acts(tool) -> LawRetriever(agent)
# LawRetriever(agent) -> Coordinator(agent)
# Coordinator(agent) -> ResponseGenerator(agent)
# ResponseGenerator(agent) -> llm.invoke(tool)
# llm.invoke(tool) -> ResponseGenerator(agent)
# ResponseGenerator(agent) -> Coordinator(agent)
# Coordinator(agent) -> User(final_answer)

# Constraints:
# - All sub-agents(query_analyzer, knowledge_retriever, law_retriever, response_generator) **MUST** work together to process the query.
# - Pipeline must be followed strictly without skipping any steps.
# - Use only the provided sub-agents for all operations.
# - Even if the necessary citations were already available to me from the `KnowledgeRetriever`'s output, I must still call the `LawRetriever` agent to ensure all relevant laws and acts are considered.
# - Ensure that each sub-agent performs its specific task without overlapping responsibilities and repeating the same task for the same query.
# - Maintain the order of operations as described in the workflow.
# - Ensure that the final answer is comprehensive and includes all relevant information from the subgraph, documents, and legal citations.
# - Ensure responses are concise, accurate, and derived solely from the knowledge graph and retrieved documents.""",
sub_agents=[query_analyzer, knowledge_retriever, law_retriever, response_generator]
)

def process_query(query: str) -> dict:
    """
    Processes a user query and returns a response.

    Args:
        query: The user's query.

    Returns:
        Coordinate the sub-agents to process the query to return a response.
    """
    relevant_subdomains_with_probs = query_analyzer.get_relevant_subdomains(query)
    relevant_subdomains = [subdomain[0] for subdomain in relevant_subdomains_with_probs]

    knowledge_data = knowledge_retriever.retrieve_knowledge(query, relevant_subdomains)
    subgraph = knowledge_data["subgraph"]
    documents = knowledge_data["documents"]

    laws_and_acts = law_retriever.get_laws_and_acts(subgraph, documents, query)
    answer = response_generator.generate_response(subgraph, documents, query, laws_and_acts)


    return {
        "answer": answer,
        "laws_and_acts": laws_and_acts
    }

root_agent = coordinator
