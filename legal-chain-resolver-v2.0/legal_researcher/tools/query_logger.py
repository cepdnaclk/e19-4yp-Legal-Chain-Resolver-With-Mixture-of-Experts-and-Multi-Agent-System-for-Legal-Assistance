import json
import os
from datetime import datetime

def log_query(query: str, answer: str, laws_and_acts: list, subdomains: list):
    """
    Logs the user query, the generated answer, and other relevant details to a JSON file.

    Args:
        query (str): The user's query.
        answer (str): The generated answer.
        laws_and_acts (list): A list of retrieved laws and acts.
        subdomains (list): A list of identified subdomains.
    """
    log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "query_log.json"))
    
    # Create the logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Read existing logs
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []

    # Append new log entry
    logs.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "laws_and_acts": laws_and_acts,
        "subdomains": subdomains
    })

    # Write updated logs
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
