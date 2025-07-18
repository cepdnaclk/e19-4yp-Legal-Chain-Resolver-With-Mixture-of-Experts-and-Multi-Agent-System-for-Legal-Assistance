from google.adk import Agent

response_generator = Agent(
    name="ResponseGenerator",
    model="gemini-2.5-flash",
    description="Generates natural language answers from a knowledge subgraph and retrieved documents.",
    instruction="""
You are a response generator agent. The coordinator provides a subgraph, relevant documents, and a list of legal citations. Your task is to generate a comprehensive natural language answer to the user's query.

Workflow:
1. Receive the subgraph, documents, query, and a list of laws and acts from the coordinator.
2. If both the subgraph and the documents are empty, ask the user for more information.
3. Otherwise, synthesize the information from the subgraph and the documents to generate a comprehensive answer.
4. At the end of your response, include a section titled 'Referred Acts and Laws' and list the provided citations as bulleted items in the following format:
   - [Act Name] No. [Act Number] of [Year], Section [Section Number]: "[Sentence/Clause]"
   - Example: ● Companies Act No. 7 of 2007, Section 529: The liquidator shall prepare a final account of the winding up.
              ● Inland Revenue Act No. 24 of 2017, Section 135: Every person chargeable to tax must furnish a return of income.
5. Return the complete answer.
6. After answering a question, return the controller to the coordinator for the next user query if there's any.
    """
)

def generate_response(subgraph: dict, documents: list, query: str, laws_and_acts: list[str]) -> str:
    """
    Generates a response from a knowledge subgraph and retrieved documents.

    Args:
        subgraph: The knowledge subgraph (nodes and edges).
        documents: A list of relevant document chunks.
        query: The user's original query.
        laws_and_acts: A list of formatted legal citations and relevant law sentences/clauses.

    Returns:
        The generated response.
    """
    if not subgraph.get('nodes') and not documents:
        return "I can't find any information about that. Can you please provide more details?"

    # Pass inputs to the agent
    result = response_generator.run({
        "query": query,
        "subgraph": subgraph,
        "documents": documents,
        "laws_and_acts": laws_and_acts
    })

    # If the result is a dict or string, handle accordingly
    if isinstance(result, str):
        return result
    elif isinstance(result, dict) and "content" in result:
        return result["content"]
    else:
        return str(result)