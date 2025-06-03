GEMINI_API_KEY = "AIzaSyDkx0sbWbFZWBy2iGR_NlusHPScQDFWRNo"
deepseek_API_KEY = "sk-6d65edba3cb94d3cba769dfcc73bc560"  

from neo4j import GraphDatabase
import google.generativeai as genai
import streamlit as st

# Define connection details
uri = "neo4j+s://6ee9af44.databases.neo4j.io:7687"  # Replace with your Neo4j URI
username = "neo4j"              # Replace with your username
password = "tj0HQRMs55aGXqSIXSWzIF_ZNDceb3zSyKo4qkTuyLM"      # Replace with your password

# Connect to the database
driver = GraphDatabase.driver(uri, auth=(username, password))
 
# Function to execute the query
def run_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters)
        return [record for record in result]

schema_query = """CALL {
  MATCH (n)
  UNWIND labels(n) AS label
  WITH label, keys(n) AS props
  UNWIND props AS prop
  RETURN label, collect(DISTINCT prop) AS properties
}
RETURN label, apoc.text.join(properties, ", ") AS properties
ORDER BY label """

Knowledge_Graph = """CREATE (diana:Person {name: "Diana"})
CREATE (melissa:Person {name: "Melissa", twitter: "@melissa"})
CREATE (dan:Person {name: "Dan", twitter: "@dan", yearsExperience: 6})
CREATE (sally:Person {name: "Sally", yearsExperience: 4})
CREATE (john:Person {name: "John", yearsExperience: 5})
CREATE (jennifer:Person {name: "Jennifer", twitter: "@jennifer", yearsExperience: 5})
CREATE (joe:Person {name: "Joe"})
CREATE (mark:Person {name: "Mark", twitter: "@mark"})
CREATE (ann:Person {name: "Ann"})
CREATE (xyz:Company {name: "XYZ"})
CREATE (x:Company {name: "Company X"})
CREATE (a:Company {name: "Company A"})
CREATE (Neo4j:Company {name: "Neo4j"})
CREATE (abc:Company {name: "ABC"})
CREATE (query:Technology {type: "Query Languages"})
CREATE (etl:Technology {type: "Data ETL"})
CREATE (integrations:Technology {type: "Integrations"})
CREATE (graphs:Technology {type: "Graphs"})
CREATE (dev:Technology {type: "Application Development"})
CREATE (java:Technology {type: "Java"})
CREATE (diana)-[:LIKES]->(query)
CREATE (melissa)-[:LIKES]->(query)
CREATE (dan)-[:LIKES]->(etl)<-[:LIKES]-(melissa)
CREATE (xyz)<-[:WORKS_FOR]-(sally)-[:LIKES]->(integrations)<-[:LIKES]-(dan)
CREATE (sally)<-[:IS_FRIENDS_WITH]-(john)-[:LIKES]->(java)
CREATE (john)<-[:IS_FRIENDS_WITH]-(jennifer)-[:LIKES]->(java)
CREATE (john)-[:WORKS_FOR]->(xyz)
CREATE (sally)<-[:IS_FRIENDS_WITH]-(jennifer)-[:IS_FRIENDS_WITH]->(melissa)
CREATE (joe)-[:LIKES]->(query)
CREATE (x)<-[:WORKS_FOR]-(diana)<-[:IS_FRIENDS_WITH]-(joe)-[:IS_FRIENDS_WITH]->(mark)-[:LIKES]->(graphs)<-[:LIKES]-(jennifer)-[:WORKS_FOR {startYear: 2017}]->(Neo4j)
CREATE (ann)<-[:IS_FRIENDS_WITH]-(jennifer)-[:IS_FRIENDS_WITH]->(mark)
CREATE (john)-[:LIKES]->(dev)<-[:LIKES]-(ann)-[:IS_FRIENDS_WITH]->(dan)-[:WORKS_FOR]->(abc)
CREATE (ann)-[:WORKS_FOR]->(abc)
CREATE (a)<-[:WORKS_FOR]-(melissa)-[:LIKES]->(graphs)<-[:LIKES]-(diana)
"""


results = run_query(schema_query)

GEMINI_API_URL = "https://api.deepmind.com/v1/chat/completions"  

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def return_cypher_query_for_user_query(userQuery):
    system_prompt = """
    You are an expert Cypher query generator for a Neo4j graph database. Your sole task is to translate natural language questions into **ONLY valid, precise, and executable Cypher queries**.

    **Your responses must strictly adhere to the following rules:**
    1.  **Strictly use the provided graph schema.** Do not infer, assume, or use any labels, properties, or relationship types not explicitly defined in the schema.
    2.  **Generate ONLY the Cypher query.** Do not include any explanations, introductory phrases, comments, or formatting blocks (like ```cypher```).
    3.  **Do not infer or include any nodes, relationships, or properties in the query that are not *directly and explicitly* requested by the user's question.** Avoid adding extra hops, filtering, or details not asked for.
    4.  **Prioritize `OPTIONAL MATCH` for relationships that might not exist for all nodes** relevant to the query, to ensure robustness and prevent filtering out results.
    5.  **Always use `COLLECT(DISTINCT property)` when the question implies gathering multiple unique values into a list** (e.g., "all interests," "who are friends," "companies worked at").
    6.  **Correctly access and filter properties on relationships** (e.g., `WHERE relationshipVar.property = value`), not on the connected node.
    7.  **Identify and use non-directional traversals for inherently bidirectional relationships.** For instance, `IS_FRIENDS_WITH` should generally be traversed as `-(:IS_FRIENDS_WITH)-` unless the question explicitly specifies a directional context.

    **This is the structure of the knowledge graph you will be querying:**
    """ + Knowledge_Graph + """

    **Here's the schema (labels, properties, and relationship types) derived from the knowledge graph for your reference:**
    """ + schema_query + """

    **EXAMPLES (Learn from these patterns):**

    **Example 1: Basic Node Property & Optional Relationship**
    QUESTION: What is Melissa’s Twitter handle and what are all the technologies she likes?
    RESPONSE:
    MATCH (m:Person {name: "Melissa"}) OPTIONAL MATCH (m)-[:LIKES]->(t:Technology) RETURN m.twitter AS TwitterHandle, COLLECT(DISTINCT t.type) AS LikedTechnologies

    **Example 2: Filtering on Relationship Property & Combining Collections**
    QUESTION: What are Jennifer’s interests, and where has she worked since 2017?
    RESPONSE:
    MATCH (jennifer:Person {name: "Jennifer"}) OPTIONAL MATCH (jennifer)-[:LIKES]->(tech:Technology) OPTIONAL MATCH (jennifer)-[worksFor:WORKS_FOR]->(company:Company) WHERE worksFor.startYear >= 2017 RETURN jennifer.name AS PersonName, COLLECT(DISTINCT tech.type) AS Interests, COLLECT(DISTINCT company.name) AS CompaniesWorkedSince2017

    **Example 3: Multiple Distinct Relationships & Robustness**
    QUESTION: What is Sally’s area of interest, and who is she connected to (friends and company)?
    RESPONSE:
    MATCH (sally:Person {name: "Sally"}) OPTIONAL MATCH (sally)-[:LIKES]->(interest:Technology) OPTIONAL MATCH (sally)-[:IS_FRIENDS_WITH]-(friend:Person) OPTIONAL MATCH (sally)-[:WORKS_FOR]->(company:Company) RETURN sally.name AS PersonName, COLLECT(DISTINCT interest.type) AS AreasOfInterest, COLLECT(DISTINCT friend.name) AS ConnectedFriends, company.name AS WorksForCompany

    ---
    **QUESTION:**

    """ + userQuery + """

    **RESPONSE:**
    """
    prompt_content = [system_prompt]
    response = model.generate_content(prompt_content,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=500,
            # stop_sequences=["\n"] # Consider if you need specific stopping for generate_content
        )
    )
    # Return the generated text
    return response.text.strip()
driver.close()


 















# user_query = "Where does Diana work, and what technologies does she like?"
# # Query to execute
# query = """
# MATCH (d:Person {name: $name})
# OPTIONAL MATCH (d)-[:WORKS_FOR]->(c:Company)
# OPTIONAL MATCH (d)-[:LIKES]->(t:Technology)
# RETURN d.name, c.name AS company, collect(t.type) AS likes
# """
# parameters = {"name": "Diana"}