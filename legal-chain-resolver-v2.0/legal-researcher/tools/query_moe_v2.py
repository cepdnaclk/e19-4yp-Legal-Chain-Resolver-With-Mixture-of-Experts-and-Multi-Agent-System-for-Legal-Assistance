import os
import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from moe_tools.expert import Expert
from moe_tools.gating import TopKGatingNetwork

script_dir = os.path.dirname(__file__)

def get_top_subdomains(query_text, top_k=4):
    """Queries the MoE system and returns the top-k subdomain names."""
    # --- Load Models and Mappings ---
    # This part can be optimized in a production system by loading models only once.
    with open(os.path.join(script_dir, "moe_tools", "subdomain_map.json"), 'r') as f:
        subdomain_map = json.load(f)
    index_to_subdomain = {i: name for name, i in subdomain_map.items()}
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    input_dim = embedding_model.get_sentence_embedding_dimension()
    num_experts = len(subdomain_map)
    gating_network = TopKGatingNetwork(input_dim, num_experts)
    gating_network.load_state_dict(torch.load(os.path.join(script_dir, "moe_tools", "gating_model.pt")))
    gating_network.eval()

    # --- Process Query ---
    query_emb = torch.tensor(embedding_model.encode(query_text)).unsqueeze(0)
    
    # Get top-k experts from gating network
    with torch.no_grad():
        _, top_k_indices = gating_network(query_emb, top_k=top_k)
    
    top_subdomains = [index_to_subdomain[idx.item()] for idx in top_k_indices[0]]
    return top_subdomains

def final_query(query_text, top_k=4):
    """Queries the MoE system and returns the top results."""
    # --- Load Models and Mappings ---
    print("Loading models and mappings...")
    # Load subdomain to index mapping
    with open(os.path.join(script_dir, "moe_tools", "subdomain_map.json"), 'r') as f:
        subdomain_map = json.load(f)
    index_to_subdomain = {i: name for name, i in subdomain_map.items()}
    
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load gating network
    input_dim = embedding_model.get_sentence_embedding_dimension()
    num_experts = len(subdomain_map)
    gating_network = TopKGatingNetwork(input_dim, num_experts)
    gating_network.load_state_dict(torch.load(os.path.join(script_dir, "moe_tools", "gating_model.pt")))
    gating_network.eval()

    # Load all expert models
    experts = {}
    for name, i in subdomain_map.items():
        expert_model = Expert(input_dim)
        expert_path = os.path.join(script_dir, "moe_tools", "subdomains", name, "expert_model.pt")
        expert_model.load_state_dict(torch.load(expert_path))
        expert_model.eval()
        experts[name] = expert_model

    # --- Process Query ---
    print(f"\nProcessing query: '{query_text}'")
    # Embed the query
    query_emb = torch.tensor(embedding_model.encode(query_text)).unsqueeze(0)
    
    # Get top-k experts from gating network
    with torch.no_grad():
        gate_probs, top_k_indices = gating_network(query_emb, top_k=top_k)
    
    print("\n--- Gating Network Results ---")
    for i in range(top_k_indices.size(1)):
        expert_index = top_k_indices[0, i].item()
        expert_name = index_to_subdomain[expert_index]
        prob = gate_probs[0, expert_index].item()
        print(f"Expert: {expert_name}, Probability: {prob:.4f}")

    # --- Retrieve from Selected Experts ---
    print("\n--- Retrieval Results ---")
    all_results = []
    
    for i in range(top_k_indices.size(1)):
        expert_index = top_k_indices[0, i].item()
        expert_name = index_to_subdomain[expert_index]
        gate_prob = gate_probs[0, expert_index].item()
        
        # Transform query embedding with the selected expert
        expert_model = experts[expert_name]
        with torch.no_grad():
            expert_query_vec = expert_model(query_emb).numpy().flatten()
        
        # Load documents for this subdomain
        subdomain_folder = os.path.join(script_dir, "moe_tools", "subdomains", expert_name)
        for fname in os.listdir(subdomain_folder):
            if fname.endswith(".embedding.npy"):
                doc_emb = torch.tensor(np.load(os.path.join(subdomain_folder, fname))).unsqueeze(0)
                with torch.no_grad():
                    expert_doc_vec = expert_model(doc_emb).numpy().flatten()
                
                # Calculate cosine similarity
                similarity = np.dot(expert_query_vec, expert_doc_vec) / (np.linalg.norm(expert_query_vec) * np.linalg.norm(expert_doc_vec))
                weighted_score = gate_prob * similarity
                
                all_results.append({
                    "subdomain": expert_name,
                    "document": fname.replace(".embedding.npy", ""),
                    "score": weighted_score
                })

    # Sort all results by final score and return top 5
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:5]

if __name__ == "__main__":
    while(True):
        print("\n--- MoE Query System ---")
        print("Type 'exit' to quit.")
        
        # Get user input
        my_query = input("Enter your query: ")
        
        if my_query.lower() == 'exit':
            print("Exiting the MoE Query System. Goodbye!")
            break
        
        top_results = final_query(my_query)

        print(f"\nTop 5 results for query: '{my_query}'")
        for result in top_results:
            print(f"  - Subdomain: {result['subdomain']}, Document: {result['document']}, Score: {result['score']:.4f}")
