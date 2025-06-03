from google.adk import Agent
import PyPDF2
import subagents.knowledge_graph as kg

full_kg_data = kg.full_kg_data
def pdf_extractor(file_path):
    try:
        # Initialize an empty string to store extracted text
        extracted_text = ""
        
        # Open the PDF file in read-binary mode
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
extracted_details = pdf_extractor("D:\Intern\FYP\kg\Agents\Legal-Assistant\legal_researcher\subagents\Relevant_Laws.pdf")

law_retriver = Agent(
    name="LawRetriver",
    model="gemini-2.0-flash-exp",
    description="Gives the relevant laws that is related to the user query.",
    instruction=f""" 
    You are a Legal Researcher for this multi agent system. The coordinator will provide you a knowledge graph , a dictionary containing 
     The coordinator provides a dictionary containing "nodes" and "edges" retrieved from the knowledge graph. Your task is to retrive what are the laws(acts or obidions,certain laws etc) that is related to the user query.
     
    WorkFlow:
    1. Receive the dictionary of nodes and edges from the coordinator.
    2. Analyze the nodes and edges and the {extracted_details} to retrive relevant relevant laws, acts, and legal principles as a lawyer would present, with references to applicable legal frameworks.
    3. Ensure the answer directly addresses the query using only the provided nodes and edges and the {extracted_details}.
    4. Return a answer giving the relevant acts, laws, and legal principles for the user query.
    
    Constraints:
    - Use only the provided nodes and edges and the {extracted_details}.
    - Ensure the answer is clear, professional, and derived solely from the input data.
    - Do not infer beyond the provided nodes and edges and the {extracted_details}.
    - Only provide if there is a law or any law related to the user query. Do not use your intuition to this.
    
    Example:
    - Input: {{
        extracted_details = "Text containing relevant laws, acts, and legal principles."
        "nodes": [{{"id": "breach of contract", "label": "breach of contract"}}, {{"id": "party fails to fulfill obligations", "label": "party fails to fulfill obligations"}}],
        "edges": [{{"source": "breach of contract", "target": "party fails to fulfill obligations", "relation": "occurs"}}]
      }}
      - Output: {{
          "answer": {{
              "Laws":list of laws related to the user query],
              "Acts":list of acts related to the user query],
              "Legal Principles":list of legal principles related to the user query]]
          }}
      }}
    """
    
    )
