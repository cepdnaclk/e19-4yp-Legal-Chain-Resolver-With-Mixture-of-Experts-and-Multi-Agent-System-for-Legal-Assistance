import os
import json
import torch
from sentence_transformers import SentenceTransformer
from moe_tools.gating import TopKGatingNetwork

# --- Load models & mappings once ---
script_dir = os.path.dirname(__file__)

with open(os.path.join(script_dir, "moe_tools", "subdomain_map.json"), 'r') as f:
    subdomain_map = json.load(f)
index_to_subdomain = {i: name for name, i in subdomain_map.items()}

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
input_dim = embedding_model.get_sentence_embedding_dimension()
num_experts = len(subdomain_map)

gating_network = TopKGatingNetwork(input_dim, num_experts)
gating_network.load_state_dict(torch.load(os.path.join(script_dir, "moe_tools", "gating_model.pt")))
gating_network.eval()


def get_top_subdomains(query_text, top_k=4):
    """
    Queries the MoE system and returns the top-k subdomains with their gating probabilities.

    Args:
        query_text (str): The user’s legal query.
        top_k (int): Number of top subdomains to return.

    Returns:
        List[List]: A list of lists: [subdomain, probability]
    """
    # Encode the query
    query_emb = torch.tensor(embedding_model.encode(query_text)).unsqueeze(0)

    # Get top-k experts from gating network
    with torch.no_grad():
        gate_probs, top_k_indices = gating_network(query_emb, top_k=top_k)

    # Prepare result
    top_results = [
        [index_to_subdomain[top_k_indices[0, i].item()],
         round(gate_probs[0, top_k_indices[0, i].item()].item(), 4)]
        for i in range(top_k_indices.size(1))
    ]

    return top_results


if __name__ == "__main__":
    while True:
        print("\n--- MoE Subdomain Query System ---")
        print("Type 'exit' to quit.")
        
        my_query = input("Enter your query: ")

        if my_query.lower() == 'exit':
            print("Exiting. Goodbye!")
            break

        results = get_top_subdomains(my_query)

        print(f"\nTop-{len(results)} subdomains for query: '{my_query}'")
        for r in results:
            print(f"[{r[0]}, {r[1]}]")
        # print(results)