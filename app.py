import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

# Page configuration
st.set_page_config(
    page_title="Advanced Langchain Ollama Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Enable Langchain debugging
set_debug(True)

# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )

# We'll use a simpler approach without streaming since Ollama in this version doesn't support it

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Chat Configuration")

    # Model selection
    model_options = ["mistral:latest", "llama2:latest", "codellama:latest", "phi3:latest"]
    selected_model = st.selectbox("Select Model", model_options)

    # Temperature slider
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # Output format selection
    output_format = st.radio("Output Format", ["Text", "JSON"])

    # System prompt customization
    system_message = st.text_area(
        "System Prompt",
        value="You are a helpful, friendly AI assistant. Provide detailed and accurate responses to the user's questions.",
        height=150
    )

    # Add a clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        st.success("Conversation cleared!")

# Main UI
st.title("🤖 Advanced Langchain Ollama Chat")
st.info("💡 This application uses Langchain with Ollama to provide an enhanced chat experience with memory and streaming responses.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to generate response
def generate_response(user_input: str) -> str:
    try:
        # Initialize the LLM with selected parameters
        llm = Ollama(
            model=selected_model,
            temperature=temperature,
        )

        # Get chat history from memory
        chat_history = st.session_state.conversation_memory.chat_memory.messages

        # Create prompt template with memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Choose output parser based on selection
        if output_format == "JSON":
            output_parser = JsonOutputParser()
        else:
            output_parser = StrOutputParser()

        # Create the chain
        chain = prompt | llm | output_parser

        # Create a container for showing progress
        with st.spinner("Generating response..."):
            # Invoke the chain without streaming
            response = chain.invoke(
                {"input": user_input, "chat_history": chat_history}
            )
            
            # Ensure we have a valid response
            if response is None:
                response = "I'm sorry, I couldn't generate a response."
            elif isinstance(response, dict):
                # If JSON response, convert to string for display
                import json
                response = json.dumps(response, indent=2)

        # Update memory with the new exchange
        st.session_state.conversation_memory.chat_memory.add_user_message(user_input)
        st.session_state.conversation_memory.chat_memory.add_ai_message(str(response))

        # Wait for LangSmith tracing to complete
        wait_for_all_tracers()

        return response

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return f"I'm sorry, I encountered an error: {str(e)}. Please try again or try a different question."

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display response
    with st.chat_message("assistant"):
        start_time = time.time()
        response = generate_response(user_input)
        end_time = time.time()
        
        # Display the response content - try both methods
        try:
            st.markdown(response)  # For text with markdown formatting
        except Exception as e:
            st.write("Error displaying with markdown:", e)
            st.write(response)  # Fallback to write
        
        # Add response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display response time
        response_time = end_time - start_time
        st.caption(f"Response generated in {response_time:.2f} seconds")

# Add footer
st.markdown("---")
st.caption("Powered by Langchain, Ollama, and Streamlit | Created with ❤️")