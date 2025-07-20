"""
Legal Researcher Agent Chat Application
=======================================

This Streamlit application provides a chat interface for interacting with the ADK Legal Researcher Agent.
It allows users to create sessions, send messages, and receive text-based legal analysis and responses.

Requirements:
------------
- ADK API Server running on localhost:8000
- Legal Researcher Agent registered and available in the ADK
- Streamlit and related packages installed

Usage:
------
1. Start the ADK API Server: `adk api_server`
2. Ensure the Legal Researcher Agent is registered and working
3. Run this Streamlit app: `streamlit run legal_researcher/legal_researcher_app.py`
4. Click "Start Legal Chain Resolver" to begin.

Architecture:
------------
- Session Management: Creates and manages ADK sessions for stateful conversations
- Message Handling: Sends user messages to the ADK API and processes responses
- Text-based Interaction: Focuses on text input and output for legal queries and analysis

API Assumptions:
--------------
1. ADK API Server runs on localhost:8000
2. Legal Researcher Agent is registered with app_name="legal_researcher"
3. Responses follow the ADK event structure with model outputs, where the output is a JSON string.

"""
import streamlit as st
import requests
import json
import uuid
import time
import re

# Set page config
st.set_page_config(
    page_title="Legal Researcher Agent Chat",
    page_icon="⚖️",
    layout="centered"
)

# Constants
API_BASE_URL = "http://localhost:8000"
APP_NAME = "legal_researcher"

# Initialize session state variables
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user-{uuid.uuid4()}"
    
if "session_id" not in st.session_state:
    st.session_state.session_id = None
    
if "messages" not in st.session_state:
    st.session_state.messages = []

def create_session():
    """
    Create a new session with the legal researcher agent.
    """
    session_id = f"session-{int(time.time())}"
    response = requests.post(
        f"{API_BASE_URL}/apps/{APP_NAME}/users/{st.session_state.user_id}/sessions/{session_id}",
        headers={"Content-Type": "application/json"},
        data=json.dumps({})
    )
    
    if response.status_code == 200:
        st.session_state.session_id = session_id
        st.session_state.messages = []
        return True
    else:
        st.error(f"Failed to create session: {response.text}")
        return False

# def send_message(message):
#     """
#     Send a message to the legal researcher agent and process the response.
#     """
#     if not st.session_state.session_id:
#         st.error("No active session. Please create a session first.")
#         return False
    
#     # Add user message to chat
#     st.session_state.messages.append({"role": "user", "content": message})
    
#     # Send message to API
#     response = requests.post(
#         f"{API_BASE_URL}/run",
#         headers={"Content-Type": "application/json"},
#         data=json.dumps({
#             "app_name": APP_NAME,
#             "user_id": st.session_state.user_id,
#             "session_id": st.session_state.session_id,
#             "new_message": {
#                 "role": "user",
#                 "parts": [{"text": message}]
#             }
#         })
#     )
    
#     if response.status_code != 200:
#         st.error(f"Error: {response.text}")
#         return False
    
#     # Process the response
#     events = response.json()
    
#     # Extract assistant's text response
#     full_response_text = ""
#     for event in events:
#         if event.get("content", {}).get("role") == "model" and "text" in event.get("content", {}).get("parts", [{}])[0]:
#             full_response_text += event["content"]["parts"][0]["text"]

#     assistant_message = None
#     if full_response_text:
#         try:
#             # The response is a JSON string, so we parse it
#             response_data = json.loads(full_response_text)
#             assistant_message = response_data.get("answer", "Could not find an answer in the response.")
#         except (json.JSONDecodeError, TypeError):
#             # If parsing fails, it might be a plain text response
#             assistant_message = full_response_text
    
#     # Add assistant response to chat
#     if assistant_message:
#         st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    
#     return True

import json
import requests
import streamlit as st

def send_message(message):
    """
    Send a message to the legal researcher agent and process the response to display only
    follow-up questions, final answers, or clarifications.
    """
    if not st.session_state.session_id:
        st.error("No active session. Please create a session first.")
        return False
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": message})
    
    # Send message to API
    try:
        response = requests.post(
            f"{API_BASE_URL}/run",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "app_name": APP_NAME,
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id,
                "new_message": {
                    "role": "user",
                    "parts": [{"text": message}]
                }
            })
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.RequestException as e:
        st.error(f"Error sending message to API: {str(e)}")
        return False
    
    # Process the response
    events = response.json()
    
    # Extract and filter assistant's text response
    assistant_message = None
    for event in events:
        content = event.get("content", {})
        if content.get("role") == "model" and content.get("parts"):
            for part in content["parts"]:
                if "text" in part:
                    try:
                        # Parse the text as JSON and extract the 'answer' field
                        response_data = json.loads(part["text"])
                        answer = response_data.get("answer", "")
                        
                        # Filter for follow-up questions, final answers, or clarifications
                        if answer.startswith("Follow-up question: "):
                            assistant_message = answer.replace("Follow-up question: ", "").strip()
                        elif answer.startswith("Clarification: "):
                            assistant_message = answer.replace("Clarification: ", "").strip()
                        elif answer.startswith("Final_Answer: "):
                            # Remove the prefix
                            cleaned_answer = answer.replace("Final_Answer: ", "").strip()
                            # Remove <b> tags from citations
                            cleaned_answer = re.sub(r'<b>(.*?)</b>', r'\1', cleaned_answer)
                            assistant_message = cleaned_answer
                        else:
                            continue  # Skip non-matching responses
                        break  # Stop once we find a valid response
                    except json.JSONDecodeError:
                        # If JSON parsing fails, check the raw text for valid prefixes
                        raw_text = part["text"]
                        if raw_text.startswith("Follow-up question: "):
                            assistant_message = raw_text.replace("Follow-up question: ", "").strip()
                        elif raw_text.startswith("Clarification: "):
                            assistant_message = raw_text.replace("Clarification: ", "").strip()
                        elif raw_text.startswith("Final_Answer: "):
                            cleaned_answer = raw_text.replace("Final_Answer: ", "").strip()
                            cleaned_answer = re.sub(r'<b>(.*?)</b>', r'\1', cleaned_answer)
                            assistant_message = cleaned_answer
                        else:
                            continue  # Skip non-matching responses
                    except TypeError:
                        assistant_message = "Invalid response format"
                if assistant_message:
                    break  # Exit the parts loop once we have a message
        if assistant_message:
            break  # Exit the event loop once we have a message
    
    # Fallback if no valid response is found
    if not assistant_message:
        assistant_message = "No valid response received from the agent."
    
    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    
    return True

if not st.session_state.session_id:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Start Legal Chain Resolver", use_container_width=True):
            create_session()
            st.rerun()
else:
    st.title("⚖️  Legal Chain Resolver  ⚖️")
    # Chat interface
    st.subheader("Ask your legal questions below")

    # Display messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # Input for new messages
    if st.session_state.session_id:
        user_input = st.chat_input("Type your legal query...")
        st.write(user_input)
        if user_input:
            send_message(user_input)
            st.rerun()
    else:
        st.info("👈 Create a session to start chatting")