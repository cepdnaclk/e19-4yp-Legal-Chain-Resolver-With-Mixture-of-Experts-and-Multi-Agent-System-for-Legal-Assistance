import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from gating import TopKGatingNetwork

class GatingDataset(Dataset):
    def __init__(self, csv_path, label_map):
        df = pd.read_csv(csv_path)
        self.emb_paths = df['query_embedding_path'].values
        self.labels = df['expert_label'].map(label_map).values

    def __len__(self):
        return len(self.emb_paths)

    def __getitem__(self, idx):
        emb = np.load(self.emb_paths[idx])
        label = self.labels[idx]
        return torch.tensor(emb), torch.tensor(label, dtype=torch.long)

def train_gating_network(csv_path="../../data/gating_train.csv", model_save_path="../../data/gating_model.pt"):
    print("Starting gating network training...")
    df = pd.read_csv(csv_path)
    # Create a consistent label map
    subdomains = sorted(df['expert_label'].unique())
    label_map = {name: i for i, name in enumerate(subdomains)}
    
    with open("../../data/subdomain_map.json", "w") as f:
        import json
        json.dump(label_map, f)

    dataset = GatingDataset(csv_path, label_map)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    sample_emb, _ = dataset[0]
    input_dim = sample_emb.shape[0]
    num_experts = len(subdomains)
    
    model = TopKGatingNetwork(input_dim, num_experts)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        for X, y in loader:
            optimizer.zero_grad()
            # In training, we use all logits for loss calculation
            logits = model.gate(X) 
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
        
    torch.save(model.state_dict(), model_save_path)
    print(f"Gating network model saved to {model_save_path}")

if __name__ == "__main__":
    train_gating_network()
