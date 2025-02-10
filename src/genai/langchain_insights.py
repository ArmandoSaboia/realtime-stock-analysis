from langchain.llms import OpenAI
from llama_index import SimpleDirectoryReader, GPTListIndex

llm = OpenAI(temperature=0.7)

def generate_insights(query):
    """
    Generate insights based on the user query.
    Here you could integrate with a language model API (e.g., using LangChain, OpenAI, etc.)
    This implementation uses LlamaIndex with GPTListIndex to load news articles from
    the "data/news_articles" directory, build an index, and query it based on the input.

    Args:
        query (str): The user query for generating insights.
        
    Returns:
        str: The generated insights or an error message.
    """
    try:
        documents = SimpleDirectoryReader("data/news_articles").load_data()
        index = GPTListIndex(documents)
        response = index.query(query)
        return response.response
    except Exception as e:
        return f"Error generating insights: {str(e)}"