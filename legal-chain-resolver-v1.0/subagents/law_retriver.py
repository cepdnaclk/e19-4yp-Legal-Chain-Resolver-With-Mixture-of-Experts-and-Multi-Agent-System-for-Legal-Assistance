from google.adk import Agent
import PyPDF2
import subagents.knowledge_graph as kg

full_kg_data = kg.full_kg_data
def pdf_extractor(file_path):
    try:
        extracted_text = ""
        
        with open(file_path, 'rb') as file:

            pdf_reader = PyPDF2.PdfReader(file)
            
            if len(pdf_reader.pages) == 0:
                raise ValueError("The PDF file is empty or contains no readable pages.")
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
        return extracted_text.strip()
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except Exception as e:
        raise ValueError(f"Error reading the PDF file: {str(e)}")
# extracted_details = pdf_extractor("D:\Intern\FYP\kg\Agents\Legal-Assistant\legal_researcher\subagents\Relevant_Laws.pdf")
extracted_details = """
Legal Analysis of Contract Law Principles

1. Essential Elements of a Valid Contract
A valid contract requires: Agreement (Restatement (Second) of Contracts § 22 (1981)); Intention to create legal relations (Edwards v. Skyways Ltd [1964] 1 WLR 349); Consensus ad Idem (Raffles v. Wichelhaus [1864] 2 H&C 906); Contractual capacity (Uniform Commercial Code § 2-204); Legality of purpose (Restatement (Second) of Contracts § 178); Possibility of performance (Taylor v. Caldwell [1863] 3 B&S 826).

2. Requirement for Contracts to Be in Writing
Under the Statute of Frauds (1677) and Uniform Commercial Code § 2-201, contracts for the sale of land, long leases, and suretyship must be in writing. The Consumer Protection Act 68 of 2008 (South Africa) regulates consumer contracts. The Law of Property Act 1925 (UK) and Consumer Credit Act 1974 (UK) also require written agreements.

3. Breach of Contract
Breach includes non-performance, partial performance, or repudiation (Hochster v. De La Tour [1853] 2 E&B 678). Governed by UNIDROIT Principles, Article 7.3.1.

4. Remedies
Remedies include specific performance (Beswick v. Beswick [1968] AC 58), damages (Hadley v. Baxendale [1854] 9 Ex 341), and cancellation.

5. Termination
Contracts end by fulfillment, mutual agreement, material breach, or impossibility (Doctrine of Frustration, Taylor v. Caldwell [1863] 3 B&S 826).
"""


law_retriver = Agent(
    name="LawRetriver",
    model="gemini-2.0-flash-exp",
    description="Extracts relevant laws, acts, and legal principles related to the user query from knowledge graph and pre-extracted PDF text.",
    instruction=f"""
You are a Legal Researcher for a multi-agent system. The coordinator provides a dictionary containing 'nodes' (list of {{'id': str, 'label': str}}), 'edges' (list of {{'source': str, 'target': str, 'relation': str}}), and {extracted_details} (str, pre-extracted PDF text). Your task is to extract relevant laws, acts, and legal principles related to the user query using the nodes, edges, and extracted_details.

WorkFlow:
1. Receive a dictionary with 'nodes', 'edges', and {extracted_details}.
2. Analyze the nodes and edges to identify contract law concepts (e.g., 'Breach of Contract', 'Specific Performance').
3. Parse extracted_details to extract laws, sentences, and legal principles using regex and keyword matching.
4. Cross-reference node/edge labels with extracted_details to ensure relevance to the query.
5. Return a dictionary with:
   - 'Laws': List of general laws (e.g., 'Common Law of Contracts').
   - 'Sentences': List of specific statutes (e.g., 'Statute of Frauds (1677)').
   - 'Legal Principles': List of principles or doctrines (e.g., 'Specific Performance').
6. If no relevant data is found, return empty lists for each category.

Constraints:
- Use only the provided nodes, edges, and {extracted_details}.
- Ensure the answer is clear, professional, and derived solely from the input data.
- Do not infer beyond the provided data.
- Use regex patterns to identify laws (e.g., 'Common Law'), acts (e.g., 'Act' followed by year or jurisdiction), and principles (e.g., doctrines, remedies).

Example:
- Input: {{
    'nodes': [{{'id': 'breach of contract', 'label': 'Breach of Contract'}}, {{'id': 'specific performance', 'label': 'Specific Performance'}}],
    'edges': [{{'source': 'breach of contract', 'target': 'specific performance', 'relation': 'remedy'}}],
    'extracted_details': 'Breach of contract governed by Uniform Commercial Code § 2-201. Remedies include Specific Performance.'
}}
- Output: {{
    'Laws': ['Common Law of Contracts'],
    'Sentences': ['Restatement (Second) of Contracts § 22 (1981)'],
    'Legal Principles': ['Specific Performance']
}}
"""
)
