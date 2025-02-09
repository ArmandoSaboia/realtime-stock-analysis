import streamlit as st
import pandas as pd
from src.genai.langchain_insights import generate_insights

# Load processed stock data
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_stock_data.csv")

# Function to predict stock trends
def predict_trend(data, query):
    avg_close = data["close"].mean()
    if "upward trend" in query.lower():
        return "Upward trend detected!" if avg_close > data["close"].iloc[-1] else "No upward trend."
    elif "downward trend" in query.lower():
        return "Downward trend detected!" if avg_close < data["close"].iloc[-1] else "No downward trend."
    else:
        return "Please ask about trends or specific stock metrics."

# Streamlit app layout
def main():
    st.title("Interactive Stock Market Analysis Dashboard 📈")

    # Sidebar for user input
    st.sidebar.header("User Interaction")
    user_query = st.sidebar.text_input("Ask a question about the stock market:", 
                                       placeholder="e.g., What are the key trends?")
    submit_button = st.sidebar.button("Submit")

    # Load data
    data = load_data()

    # Display stock data chart
    st.subheader("Stock Price Over Time")
    st.line_chart(data.set_index("date")["close"])

    # Handle user queries
    if submit_button and user_query:
        st.subheader("AI-Generated Insights")
        with st.spinner("Generating insights..."):
            insights = generate_insights(user_query)
            st.write(insights)

        # Predict trends based on user query
        st.subheader("Trend Prediction")
        with st.spinner("Predicting trends..."):
            trend_prediction = predict_trend(data, user_query)
            st.write(trend_prediction)

    # Additional visualizations
    st.subheader("Key Metrics")
    st.metric("Average Close Price", round(data["close"].mean(), 2))
    st.metric("Max High Price", round(data["high"].max(), 2))
    st.metric("Min Low Price", round(data["low"].min(), 2))

if __name__ == "__main__":
    main()