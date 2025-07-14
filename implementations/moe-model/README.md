# Legal MoE AI: A Mixture of Experts System for Semantic Legal Document Retrieval

## 1. Project Overview

This project implements a sophisticated Mixture of Experts (MoE) model designed for the intelligent retrieval of legal documents. By leveraging specialized "expert" models for different subdomains of law (e.g., Company Law, Tax Law, Banking Law), the system can provide highly relevant search results from a large corpus of legal texts. A top-k gating network routes user queries to the most appropriate experts, ensuring both accuracy and efficiency.

### Key Features

* **Modular Architecture:** Each legal domain is handled by a dedicated expert model, making the system scalable and easy to maintain.
* **Semantic Search:** Utilizes state-of-the-art sentence transformers to understand the meaning behind legal queries, going beyond simple keyword matching.
* **Intelligent Routing:** A trainable top-k gating network dynamically selects the most relevant expert(s) for a given query, improving retrieval accuracy.
* **End-to-End Pipeline:** Includes comprehensive scripts for data preparation, text extraction, model training, and final inference.
* **Similarity-Based Learning:** Experts are trained using contrastive loss to learn meaningful representations, enabling them to identify semantically similar legal clauses.


## 2. Project Structure

The project is organized into a modular structure to separate data, source code, and notebooks.

```
legal-moe-project/
├── data/                      # All datasets and *generated* model outputs and intermediate files (ignored by Git)
│   ├── extracted_text/        # Processed data, using original pdf files (generated, ignored by Git)
│   ├── gating_queries/        # .npy embeddings of queries used to train the gating network (generated, ignored by Git)
│   ├── splitting-pdfs/        # Original unsplitted pdfs and python scripts which were used to split them (generated, ignored by Git)
│   ├── raw_pdfs/              # Original splitted legal PDFs, organized into sections (generated, ignored by Git)
│   ├── subdomains/            # Processed data, organized by legal subdomain (generated, ignored by Git)
│   │   ├── company_law/           # Example subdomain folder
│   │   │   ├── *.cleaned.txt          # Cleaned text sections (generated, ignored by Git)
│   │   │   ├── *.embedding.npy        # Vector embeddings of sections (generated, ignored by Git)
│   │   │   ├── pairs.csv              # Positive/negative training pairs for expert (generated, ignored by Git)
│   │   │   ├── faiss.index            # FAISS index for similarity search (generated, ignored by Git)
│   │   │   ├── faiss_files.pkl        # Mapping of files in FAISS index (generated, ignored by Git)
│   │   │   └── expert_model.pt        # Trained expert model weights (generated, ignored by Git)
│   │   └── ...                    # Other legal subdomains follow similar structure
│   ├── gating_model.pt        # Trained weights of the gating network (generated, ignored by Git)
│   ├── gating_train.csv       # Training data for gating network (queries & labels) (generated, ignored by Git)
│   └── subdomain_map.json     # Maps human-readable subdomain names to labels
│
├── notebooks/                 # Jupyter notebooks for data preparation and exploration
│   ├── pdf_section_extraction.ipynb # Extract & generate text from PDFs
│   ├── preprocessing.ipynb          # Clean the text extracted by removing white space, etc and mapping each text files to relevent subdomain
│   ├── embed_sections.ipynb         # Generate embeddings from cleaned text
│   └── build_faiss_and_pairs.ipynb  # Build FAISS indices & generate pairs
│
├── src/                       # Source code
│   ├── experts/                 # Expert model components
│   │   ├── expert.py                # Expert model architecture
│   │   ├── contrastive_loss.py      # Contrastive loss definition
│   │   └── train_experts.py         # Script to train all expert models
│   ├── moe/                     # Mixture of Experts components
│   │   ├── gating.py                # Gating network architecture
│   │   ├── train_gating_network.py  # Script to train gating network
│   │   └── prepare_gating_data.py   # Script to create gating_train.csv
│   └── utils.py                 # Utility functions
│
├── tests/                     # Unit and integration tests
│
├── query_moe.py               # Script to run a query through the trained MoE system
├── README.md                  # Project documentation (this file)
└── requirements.txt           # Python dependencies
```


## 3. Setup and Installation

Follow these steps to set up the project environment.

### 3.1. Clone the Repository

```
git clone <your-repository-url>
cd legal-moe-project
```


### 3.2. Create and Activate a Virtual Environment

Create the environment:

```
python -m venv venv
```

Activate it (on Windows):

```
venv\Scripts\activate
```

Activate it (on macOS/Linux):

```
source venv/bin/activate
```


### 3.3. Install Dependencies

First, create a `requirements.txt` file with the following content:

```
torch
pandas
scikit-learn
networkx
spacy
flask
jupyter
pdfplumber
sentence-transformers
fpdf
numpy
pypdf2
faiss-cpu
```

Then, install the packages:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```


## 4. End-to-End Workflow

### Step 1: Data Preparation

1. **Place PDFs:** Put your original, unsplit PDF documents in `data/splitting-pdfs/` and your pre-sectioned PDFs into the `data/raw_pdfs/` directory. (Note: These directories are for input and generated files, and are ignored by Git.)
2. **Extract Text:** Use the `notebooks/pdf_section_extraction.ipynb` to process the PDFs. This notebook reads PDFs from `data/raw_pdfs/` and saves the raw text into `data/extracted_text/`. (Note: `data/extracted_text/` contains generated files and is ignored by Git.)
3. **Preprocess and Organize Text:** Run `notebooks/preprocessing.ipynb`. This script cleans the text from the previous step and organizes the `.cleaned.txt` files into the appropriate folders under `data/subdomains/`. (Note: `data/subdomains/` contains generated files and is ignored by Git.)

### Step 2: Generate Embeddings and Build Search Indices

1. **Generate Embeddings:** Run `notebooks/embed_sections.ipynb`. This notebook iterates through all `.cleaned.txt` files in `data/subdomains/`, generates a sentence-transformer embedding for each, and saves it as a `.embedding.npy` file. (Note: These `.npy` files are generated and ignored by Git.)
2. **Build FAISS Index:** Run `notebooks/build_faiss_and_pairs.ipynb`. This creates a FAISS index for each subdomain for fast similarity searches and saves it as `faiss.index`. (Note: These index files are generated and ignored by Git.)

### Step 3: Create Training Pairs for Experts

The `notebooks/build_faiss_and_pairs.ipynb` also handles this step. It uses the FAISS index to find semantically similar and dissimilar pairs of sections within each subdomain and saves them to a `pairs.csv` file, which is crucial for training the experts. (Note: These `pairs.csv` files are generated and ignored by Git.)

### Step 4: Train the Expert Models

The expert models are trained to understand the nuances of their specific legal domain.

* **Run Training Script:**

```
python src/experts/train_experts.py
```

* **Output:** This script trains an expert model for each subdomain and saves the trained weights as `expert_model.pt` inside each subdomain's folder. (Note: These model files are generated and ignored by Git.)


### Step 5: Train the Gating Network

The gating network learns to route a user's query to the most relevant expert(s).

1. **Prepare Training Data:** Run the `src/moe/prepare_gating_data.py` script. This automatically generates the `data/gating_train.csv` file and query embeddings required for training. (Note: These files are generated and ignored by Git.)
2. **Run the Training Script:**

```
python src/moe/train_gating_network.py
```


* **Output:** This saves the trained model to `data/gating_model.pt` and a label mapping to `data/subdomain_map.json`. (Note: `gating_model.pt` is generated and ignored by Git.)


## 5. Running the MoE System for Inference

Once all models are trained, you can query the system using the main inference script.

* **How to Run:**

```
python query_moe.py
```

* **Functionality:** This script starts an interactive session in your terminal. It will prompt you to enter a query. The script then loads the gating_model.pt and all expert models, routes your query to the top-k experts, retrieves the most relevant documents, and prints the top 5 final weighted results to the console. You can enter new queries repeatedly. Type exit to quit the program.

