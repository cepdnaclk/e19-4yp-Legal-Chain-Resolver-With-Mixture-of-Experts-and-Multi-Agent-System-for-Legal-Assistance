import os
import numpy as np
import faiss
import pickle
import pandas as pd

BASE_DIR = "../data/subdomains/"

def build_faiss_indices():
    """Builds a FAISS index for each subdomain for fast similarity search."""
    print("Building FAISS indices...")
    subdomains = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

    for subdomain in subdomains:
        folder = os.path.join(BASE_DIR, subdomain)
        embeddings = []
        files = []
        for fname in os.listdir(folder):
            if fname.endswith('.embedding.npy'):
                emb = np.load(os.path.join(folder, fname))
                embeddings.append(emb)
                files.append(fname)
        
        if not embeddings:
            print(f"[WARNING] No embeddings found for {subdomain}. Skipping FAISS index.")
            continue

        embeddings_np = np.stack(embeddings)
        index = faiss.IndexFlatL2(embeddings_np.shape[1])
        index.add(embeddings_np)
        
        faiss.write_index(index, os.path.join(folder, 'faiss.index'))
        with open(os.path.join(folder, 'faiss_files.pkl'), 'wb') as f:
            pickle.dump(files, f)
    print("FAISS indices built.")

def create_similarity_pairs(k=3, num_neg=3):
    """Creates positive and negative training pairs using semantic similarity."""
    print("Creating similarity pairs for expert training...")
    subdomains = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

    for subdomain in subdomains:
        folder = os.path.join(BASE_DIR, subdomain)
        index_path = os.path.join(folder, 'faiss.index')
        
        if not os.path.exists(index_path):
            print(f"[WARNING] FAISS index not found for {subdomain}. Skipping pair creation.")
            continue

        index = faiss.read_index(index_path)
        with open(os.path.join(folder, 'faiss_files.pkl'), 'rb') as f:
            files = pickle.load(f)
        
        embeddings = [np.load(os.path.join(folder, fname)) for fname in files]
        pairs = []

        for i, emb in enumerate(embeddings):
            # Positive pairs
            D, I = index.search(np.expand_dims(emb, axis=0), k + 1)
            for j in range(1, k + 1):
                idx = I[0][j]
                pairs.append({
                    'file1': os.path.abspath(os.path.join(folder, files[i])),
                    'file2': os.path.abspath(os.path.join(folder, files[idx])),
                    'label': 1
                })
            
            # Negative pairs
            neg_indices = list(set(range(len(files))) - set(I[0]))
            np.random.shuffle(neg_indices)
            for j in range(min(num_neg, len(neg_indices))):
                idx = neg_indices[j]
                pairs.append({
                    'file1': os.path.abspath(os.path.join(folder, files[i])),
                    'file2': os.path.abspath(os.path.join(folder, files[idx])),
                    'label': 0
                })
        
        df = pd.DataFrame(pairs)
        df.to_csv(os.path.join(folder, 'pairs.csv'), index=False)
    print("Similarity pairs created.")

