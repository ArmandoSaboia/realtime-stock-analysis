from langchain.llms import OpenAI
from llama_index import SimpleDirectoryReader, GPTListIndex

llm = OpenAI(temperature=0.7)

def generate_insights(query):
    try:
        documents = SimpleDirectoryReader("data/news_articles").load_data()
        index = GPTListIndex(documents)
        response = index.query(query)
        return response.response
    except Exception as e:
        return f"Error generating insights: {str(e)}"