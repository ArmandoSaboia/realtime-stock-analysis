import streamlit as st
from llama_index.core import SimpleDirectoryReader, GPTListIndex
from langchain_groq import ChatGroq
from src.genai.config import config

def generate_insights(query):
    # For now, we'll use a dummy directory reader. In a real scenario, this would load actual data.
    # reader = SimpleDirectoryReader(input_dir="./data")
    # documents = reader.load_data()

    # For demonstration, let's assume some dummy documents
    documents = ["Stock market showed a bullish trend today.", "Technology stocks are on the rise."]

    groq_api_key = config.get("groq", {}).get("api_key")

    if not groq_api_key:
        raise ValueError("No API key found for Groq. Please set GROQ_API_KEY environment variable.")

    llm = ChatGroq(temperature=0.7, groq_api_key=groq_api_key)

    # Placeholder for actual insight generation using llama_index and LLM
    # In a real application, you would build an index from documents and query it.
    # For now, we'll just use the LLM to respond to the query.
    response = llm.invoke(f"Generate insights based on the following query: {query}. Consider these documents: {documents}")
    return response.content
