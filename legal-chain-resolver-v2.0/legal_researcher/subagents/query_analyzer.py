from google.adk import Agent
from legal_researcher.tools.query_moe_v2 import get_top_subdomains

query_analyzer = Agent(
    name="QueryAnalyzer",
    model="gemini-2.5-flash",
    description="Uses a Mixture-of-Experts model to identify the most relevant legal subdomains for a given user query.",
    instruction=f"""
You are a query analyzer agent. The coordinator provides a user query. Your task is to determine which legal subdomains are most relevant to it by calling the MoE system.

Workflow:
1. Receive the user query from the coordinator.
2. Call the `get_relevant_subdomains` function from the MoE system with the query.
3. Return a list of the most relevant subdomains and their probabilities as [subdomain_name, probability] pairs.
4. If no relevant subdomains are found, return an empty list.

Constraints:
- Use only the `get_relevant_subdomains` function to determine subdomains.
- Do not guess or fabricate subdomains.
- Ensure the subdomains returned are accurate and directly relevant to the query.

Examples:
- Input: Query = "What happens if someone uses my idea without permission?"
  - Output: [['ip_law', 0.6672], ['contract_law', 0.1125], ['electronic_transactions_law', 0.1121], ['securities_law', 0.1082]]
- Input: Query = "An employee leaves a firm and starts a competing business, using designs and materials from the former company. How should the original company respond?"
  - Output: [['company_law', 0.5555], ['consumer_law', 0.1783], ['ip_law', 0.1695], ['securities_law', 0.0967]]
"""
)

def get_relevant_subdomains(query: str) -> list[list]:
    """
    Identifies and returns the most relevant legal subdomains and their probabilities
    by invoking the MoE system.

    Args:
        query (str): The user's legal query.

    Returns:
        list[list]: A list of [subdomain_name: str, probability: float] pairs,
                    or an empty list if no subdomains are identified.

    Example:
        >>> get_relevant_subdomains("How do I register a trademark?")
        [['ip_law', 0.6532], ['company_law', 0.3468]]
    """
    return get_top_subdomains(query)