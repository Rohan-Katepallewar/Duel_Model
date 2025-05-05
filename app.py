import streamlit as st
import os
from openai import OpenAI
import google.generativeai as genai
import time

# Set page configuration
st.set_page_config(
    page_title="AI Chatbot with Summary",
    page_icon="💬",
    layout="wide"
)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_ended" not in st.session_state:
    st.session_state.chat_ended = False

if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "sentiment" not in st.session_state:
    st.session_state.sentiment = ""

# Sidebar for API keys
with st.sidebar:
    st.title("API Configuration")
    google_api_key = st.text_input("Google API Key", type="password")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot uses Gemini for conversations and OpenAI for summarization and sentiment analysis when the chat ends.")

# Main title
st.title("🤖 Multi-Model Chatbot")

# Function to initialize Gemini
def initialize_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

# Function to get response from Gemini
def get_gemini_response(model, user_input):
    try:
        # Include conversation history for context
        full_context = ""
        for msg in st.session_state.messages:
            role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
            full_context += f"{role_prefix}{msg['content']}\n\n"
        
        full_context += f"User: {user_input}\n\nAssistant: "
        
        response = model.generate_content(full_context)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Function to generate summary and sentiment analysis using OpenAI
def generate_summary_and_sentiment(api_key):
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare conversation history for OpenAI
        conversation_text = ""
        for msg in st.session_state.messages:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            conversation_text += f"{prefix}{msg['content']}\n\n"
        
        # Create prompt for OpenAI
        system_prompt = """
        You are an expert conversation analyst. You need to provide two separate outputs:
        
        1. A clear and concise summary of the conversation (approximately 150 words)
        2. A brief sentiment analysis of the conversation
        
        Format your response exactly as follows, maintaining the exact headings:
        
        ###SUMMARY###
        [Your 150-word summary here]
        
        ###SENTIMENT###
        [Your brief sentiment analysis here, indicating if the conversation was positive, negative, or neutral, with a brief explanation]
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ]
        )
        
        # Extract the response text
        full_response = response.choices[0].message.content
        
        # Extract summary and sentiment using the markers
        try:
            summary_part = full_response.split("###SUMMARY###")[1].split("###SENTIMENT###")[0].strip()
            sentiment_part = full_response.split("###SENTIMENT###")[1].strip()
        except IndexError:
            # Fallback if the format wasn't exactly followed
            parts = full_response.split("\n\n")
            if len(parts) >= 2:
                summary_part = parts[0]
                sentiment_part = parts[1]
            else:
                summary_part = full_response
                sentiment_part = "Sentiment analysis unavailable."
        
        return summary_part, sentiment_part
    except Exception as e:
        st.error(f"Error generating analysis: {str(e)}")
        return "Error generating summary.", f"Error: {str(e)}"

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input area when chat is not ended
if not st.session_state.chat_ended:
    # Check if we have the Google API key
    if not google_api_key:
        st.warning("Please enter your Google API key in the sidebar to start chatting.")
    else:
        # Initialize Gemini model
        gemini_model = initialize_gemini(google_api_key)
        
        if gemini_model:
            # Get user input
            user_input = st.chat_input("Type your message here...")
            
            if user_input:
                # Add user message to history and display it
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Get and display Gemini's response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking...")
                    
                    response = get_gemini_response(gemini_model, user_input)
                    message_placeholder.markdown(response)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Only show End Chat button if there are messages
            if st.session_state.messages:
                if st.button("End Chat", key="end_chat"):
                    st.session_state.chat_ended = True
                    st.experimental_rerun()

# Handle end of chat - generate summary and sentiment analysis
elif st.session_state.chat_ended and not st.session_state.summary_generated:
    # Check if we have the OpenAI API key
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to generate the summary and sentiment analysis.")
        if st.button("Resume Chat", key="resume_chat"):
            st.session_state.chat_ended = False
            st.experimental_rerun()
    else:
        # Show a spinner while generating the summary and sentiment
        with st.spinner("Generating conversation summary and sentiment analysis..."):
            # Generate summary and sentiment analysis
            summary, sentiment = generate_summary_and_sentiment(openai_api_key)
            
            # Store in session state
            st.session_state.summary = summary
            st.session_state.sentiment = sentiment
            st.session_state.summary_generated = True
            
            # Rerun to display the summary and sentiment
            st.experimental_rerun()

# Display summary and sentiment analysis when generated
elif st.session_state.summary_generated:
    # Display the summary
    st.subheader("Conversation Summary")
    st.write(st.session_state.summary)
    
    # Display the sentiment analysis
    st.subheader("Sentiment Analysis")
    st.write(st.session_state.sentiment)
    
    # Option to start a new chat
    if st.button("Start New Chat", key="new_chat"):
        # Reset all session states
        st.session_state.messages = []
        st.session_state.chat_ended = False
        st.session_state.summary_generated = False
        st.session_state.summary = ""
        st.session_state.sentiment = ""
        st.experimental_rerun()