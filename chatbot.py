# Install required libraries (Run this in your terminal or command line)
# pip install streamlit transformers torch

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Hardcoded responses for basic questions about AI
def get_predefined_response(user_input):
    if "what is artificial intelligence" in user_input.lower():
        return ("Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed "
                "to think like humans and mimic their actions. AI can be categorized into two types: narrow AI, which is "
                "designed for specific tasks, and general AI, which has the ability to understand and reason across a wide range of tasks.")
    elif "explain" in user_input.lower():
        return ("Artificial Intelligence encompasses various technologies, including machine learning, natural language processing, "
                "and robotics. These technologies enable machines to perform tasks that typically require human intelligence, such as "
                "recognizing speech, understanding language, and making decisions.")
    return None

# Function to generate a response
def generate_response(user_input, chat_history_ids):
    # Encode the new user input and append it to the chat history
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Concatenate the new input with the chat history
    if chat_history_ids is not None:
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        chat_history_ids = new_user_input_ids

    # Generate a response
    bot_output = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response and extract the chatbot's reply
    bot_response = tokenizer.decode(bot_output[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

    return bot_response, chat_history_ids

# Streamlit app layout
st.title("AI Chatbot")
st.write("Hi! I'm here to help you understand artificial intelligence. Type your question below.")

# Session state for chat history
if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None

# Text input for user
user_input = st.text_input("You: ")

# Display chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if user_input:
    # Check for predefined responses first
    predefined_response = get_predefined_response(user_input)
    if predefined_response:
        st.session_state.chat_history.append(f"AI Chatbot: {predefined_response}")
    else:
        # Generate a response from the chatbot
        bot_response, st.session_state.chat_history_ids = generate_response(user_input, st.session_state.chat_history_ids)
        st.session_state.chat_history.append(f"AI Chatbot: {bot_response}")

# Show the chat history
for response in st.session_state.chat_history:
    st.write(response)

# Run the app using:
# streamlit run your_script_name.py
