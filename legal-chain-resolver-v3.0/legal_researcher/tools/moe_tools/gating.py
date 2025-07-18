import torch
import torch.nn as nn

class TopKGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x, top_k=2):
        logits = self.gate(x)
        # Get top-k experts and their logits
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        # Apply softmax to the logits of the top-k experts
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        # Create a sparse tensor for the final probabilities
        zeros = torch.zeros_like(logits, requires_grad=True)
        return zeros.scatter(1, top_k_indices, top_k_probs), top_k_indices
