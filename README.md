# Ollama-intro
# 🤖 Advanced LangChain Ollama Chat with Streamlit

This project is a **learning-focused application** designed to explore and understand how to:

- Integrate Large Language Models (LLMs) via **Ollama**
- Implement conversational logic using **LangChain**
- Leverage **LangSmith** for tracing and observability

> ⚠️ **Note:** This project was built purely for educational purposes to experiment with LLMs, LangChain memory, and LangSmith integrations.

---

## 🚀 Features

✅ Choose from multiple Ollama-supported models  
✅ Customize temperature, system prompt, and output format  
✅ Uses `ConversationBufferMemory` for contextual memory  
✅ JSON parsing with `JsonOutputParser`  
✅ Real-time response tracking with LangSmith + debugging  
✅ Clear chat history anytime  
✅ Clean, interactive chat UI with response timers

---

## 📦 Requirements

Make sure you have the following:

- Python 3.11
- [Ollama](https://ollama.com) installed and running
- A supported Ollama model pulled (e.g. `mistral`, `phi3`, etc.)
- LangChain dependencies
- LangSmith account (optional, for tracing)

---

## 🛠️ Installation

```bash
# Create a new Python environment
conda create -n langchain-chat python=3.11
conda activate langchain-chat

# Install dependencies
pip install -r requirements.txt
