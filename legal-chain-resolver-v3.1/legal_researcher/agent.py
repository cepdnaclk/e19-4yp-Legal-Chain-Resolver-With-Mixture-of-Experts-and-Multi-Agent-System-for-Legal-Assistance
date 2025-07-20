from google.adk.agents import SequentialAgent
from legal_researcher.subagents.query_analyzer import query_analyzer
from legal_researcher.subagents.knowledge_retriever import knowledge_retriever
from legal_researcher.subagents.response_generator import response_generator
from legal_researcher.subagents.law_retriever import law_retriever

# Coordinator Agent
coordinator = SequentialAgent(
    name="Coordinator",
    description="Coordinates query processing for legal questions.",
    sub_agents=[query_analyzer, knowledge_retriever, law_retriever, response_generator]
)

# --- Stateful Conversation Management ---
conversation_state = {
    "history": [],
    "subgraph": None,
    "documents": None,
    "laws_and_acts": None,
    "context_stored": False
}

def process_query(query: str) -> dict:
    """
    Processes a user query and returns a response, managing a two-step conversation.

    Args:
        query: The user's query.

    Returns:
        A dictionary containing the response and a flag indicating if it is the final answer.
    """
    global conversation_state
    conversation_state["history"].append({"role": "user", "content": query})

    # If context is not stored, this is the first turn.
    if not conversation_state["context_stored"]:
        # Run the full analysis pipeline and store the context.
        relevant_subdomains_with_probs = query_analyzer.get_relevant_subdomains(query)
        relevant_subdomains = [subdomain[0] for subdomain in relevant_subdomains_with_probs]

        knowledge_data = knowledge_retriever.retrieve_knowledge(query, relevant_subdomains)
        conversation_state["subgraph"] = knowledge_data["subgraph"]
        conversation_state["documents"] = knowledge_data["documents"]

        conversation_state["laws_and_acts"] = law_retriever.get_laws_and_acts(
            subgraph=conversation_state["subgraph"],
            documents=conversation_state["documents"],
            query=query
        )
        conversation_state["context_stored"] = True

    # Generate a response using the stored context and the full history.
    answer = response_generator.generate_response(
        subgraph=conversation_state["subgraph"],
        documents=conversation_state["documents"],
        query=query,  # Pass the latest user message for immediate context
        laws_and_acts=conversation_state["laws_and_acts"],
        conversation_history=conversation_state["history"]
    )

    conversation_state["history"].append({"role": "assistant", "content": answer})

    # A response is considered final if it doesn't end with a question mark.
    is_final = not answer.strip().endswith("?")

    if is_final:
        final_laws = conversation_state["laws_and_acts"]
        # Reset state for the next independent query.
        conversation_state = {
            "history": [], "subgraph": None, "documents": None, "laws_and_acts": None, "context_stored": False
        }
        return {
            "answer": answer,
            "laws_and_acts": final_laws,
            "is_final": True
        }
    else:
        # It's a list of questions, so the conversation continues.
        return {
            "answer": answer,
            "is_final": False
        }

root_agent = coordinator