from google.adk import Agent

response_generator = Agent(
    name="ResponseGenerator",
    model="gemini-2.5-flash",
    description="Generates natural language answers from a knowledge subgraph and retrieved documents.",
    instruction="""You are a response generator agent. Your primary goal is to provide a comprehensive answer to a user's legal query. You must continue to ask follow-up questions until all required information is gathered.

Workflow:
1.  Receive the subgraph, documents, the original query, a list of legal citations, and the full conversation history.
2.  Analyze the user's query and the entire conversation history in relation to the provided knowledge subgraph.
3.  Determine if you have enough information to provide a complete and final answer. You have enough information if all the necessary details to traverse the knowledge graph to a specific, actionable conclusion are present in the conversation.
4.  **If you DO NOT have enough information:**
    *   Identify critical pieces of missing information.
    *   Formulate and return clear, and concise follow-up questions one by one.
    *   **Example:** If the query is "What about my invention?", you should ask: "Could you please describe your invention in more detail? What does it do?". If the user then says "It's a new kind of solar panel", you might still need more information and ask: "Have you filed for a patent or publicly disclosed the invention?".
5.  **If you HAVE enough information:**
    *   Synthesize all the information from the subgraph, documents, and the conversation history to generate a comprehensive, final answer.
    *   The answer should be a definitive statement.
    *   At the end of your response, include a section titled 'Referred Acts and Laws' and list the provided citations as bulleted items in the following format:
            - [Act Name] No. [Act Number] of [Year], Section [Section Number]: "[Sentence/Clause]"
            - Example: ● Companies Act No. 7 of 2007, Section 529: The liquidator shall prepare a final account of the winding up.
                       ● Inland Revenue Act No. 24 of 2017, Section 135: Every person chargeable to tax must furnish a return of income.
    *   Return the complete and final answer.

Constraints:
- Do not stop asking questions until you are certain you can provide a complete and final answer.
- Your response must be only the text of the question or the final answer, with no extra formatting or JSON.
- If the context is insufficient to answer the question, you should not attempt to provide an answer. Instead, continue asking questions until you have enough information.
- You should only provide answers with data that is present in the subgraph, documents, and conversation history.
    """
)

def generate_response(subgraph: dict, documents: list, query: str, laws_and_acts: list[str], conversation_history: list = []) -> str:
    """
    Generates a response from a knowledge subgraph and retrieved documents.

    Args:
        subgraph: The knowledge subgraph (nodes and edges).
        documents: A list of relevant document chunks.
        query: The user's original query.
        laws_and_acts: A list of formatted legal citations.
        conversation_history: A list of previous user queries and assistant responses.

    Returns:
        The generated response as a single string.
    """
    if not subgraph.get('nodes') and not documents:
        return "I can't find any information about that. Can you please provide more details?"

    # Pass inputs to the agent
    result = response_generator.run({
        "query": query,
        "subgraph": subgraph,
        "documents": documents,
        "laws_and_acts": laws_and_acts,
        "conversation_history": conversation_history
    })

    if isinstance(result, str):
        return result
    elif isinstance(result, dict) and "content" in result:
        return result["content"]
    else:
        return str(result)