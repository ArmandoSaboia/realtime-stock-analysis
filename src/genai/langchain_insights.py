import os
from langchain.llms import OpenAI
from llama_index.readers import SimpleDirectoryReader
from llama_index import GPTListIndex
from src.genai.config import config

# Retrieve the OpenAI API key from the configuration
openai_api_key = config.get("openai", {}).get("api_key")
if not openai_api_key:
    raise ValueError("The OpenAI API key is not set. Please add it to config/config.yaml or set the environment variable.")

# Instantiate the OpenAI LLM
llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

def generate_insights(query):
    """
    Generate insights from news articles using the OpenAI model.
    Loads documents from 'data/news_articles', creates an index,
    converts it to a query engine, and queries it.
    """
    try:
        documents = SimpleDirectoryReader("data/news_articles").load_data()
        index = GPTListIndex(documents)
        # Convert index to a query engine
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return response.response
    except Exception as e:
        return f"Error generating insights: {str(e)}"