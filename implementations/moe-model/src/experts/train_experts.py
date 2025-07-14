import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from expert import Expert
from contrastive_loss import ContrastiveLoss

class PairDataset(Dataset):
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        
        self.pairs = []
        self.labels = []
        for _, row in df.iterrows():
            path1 = os.path.normpath(row['file1'])
            path2 = os.path.normpath(row['file2'])
            if os.path.exists(path1) and os.path.exists(path2):
                self.pairs.append((path1, path2))
                self.labels.append(row['label'])
        
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2 = self.pairs[idx]
        emb1 = np.load(path1)
        emb2 = np.load(path2)
        label = self.labels[idx]
        return torch.tensor(emb1), torch.tensor(emb2), torch.tensor(label)

def train_all_experts(base_dir="../../data/subdomains/"):
    print("Starting expert model training...")
    subdomains = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdomain in subdomains:
        print(f"\n--- Training expert for: {subdomain} ---")
        folder = os.path.join(base_dir, subdomain)
        csv_path = os.path.join(folder, 'pairs.csv')
        
        try:
            dataset = PairDataset(csv_path)
            if len(dataset) == 0:
                print(f"[WARNING] No valid pairs found for {subdomain}. Skipping training.")
                continue

            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            sample_emb, _, _ = dataset[0]
            input_dim = sample_emb.shape[0]
            
            model = Expert(input_dim)
            criterion = ContrastiveLoss(margin=1.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            for epoch in range(10):
                running_loss = 0.0
                for emb1, emb2, label in loader:
                    optimizer.zero_grad()
                    out1, out2 = model(emb1), model(emb2)
                    loss = criterion(out1, out2, label)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                avg_loss = running_loss / len(loader)
                print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")

            torch.save(model.state_dict(), os.path.join(folder, "expert_model.pt"))
            print(f"Model saved for {subdomain}.")

        except Exception as e:
            print(f"[ERROR] Failed to train expert for {subdomain}: {e}")
            
    print("\nExpert training complete.")

if __name__ == "__main__":
    train_all_experts()
