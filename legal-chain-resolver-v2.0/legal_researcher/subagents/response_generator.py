from google.adk import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

response_generator = Agent(
    name="ResponseGenerator",
    model="gemini-2.5-flash",
    description="Generates natural language answers from a knowledge subgraph and retrieved documents.",
    instruction="""
You are a response generator agent. The coordinator provides a subgraph and relevant documents. Your task is to generate a comprehensive natural language answer to the user's query.

Workflow:
1. Receive the subgraph (nodes and edges) and the retrieved documents from the coordinator.
2. If both the subgraph and the documents are empty, ask the user for more information.
3. Otherwise, synthesize the information from the subgraph and the documents to generate a comprehensive answer.
4. Return the answer.
    """
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def generate_response(subgraph: dict, documents: list, query: str) -> str:
    """
    Generates a response from a knowledge subgraph and retrieved documents.

    Args:
        subgraph: The knowledge subgraph (nodes and edges).
        documents: A list of relevant document chunks.
        query: The user's original query.

    Returns:
        The generated response.
    """
    if not subgraph['nodes'] and not documents:
        return "I can't find any information about that. Can you please provide more details?"

    # Combine the subgraph and documents into a single prompt for the LLM
    prompt = f"""
    Here is some information from a knowledge graph:
    {subgraph}

    Here is some additional context from retrieved documents:
    {documents}

    Based on all of this information, please answer the following question:
    {query}
    """

    response = llm.invoke(prompt)
    return response.content
