import pandas as pd
from neo4j import GraphDatabase
from send_to_llm import return_cypher_query_for_user_query
import streamlit as st

# Define connection details
uri = "neo4j+s://6ee9af44.databases.neo4j.io:7687"  
username = "neo4j"             
password = "tj0HQRMs55aGXqSIXSWzIF_ZNDceb3zSyKo4qkTuyLM"      

# Connect to the database
driver = GraphDatabase.driver(uri, auth=(username, password))

# Function to execute the query
def run_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters)
        return [record for record in result]

st.title("Neo4j Query Interface")

userQuery = st.text_input("Enter your user userQuery:")

if userQuery:
    st.write(f"Processing userQuery: {userQuery}")

    cypher_query = return_cypher_query_for_user_query(userQuery)
    results = run_query(cypher_query)
    print(cypher_query)
    print(results)
    
    st.subheader("Answer using the neo4j KG:")
    
    st.write("Answer",results)