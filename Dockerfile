FROM python:3.9-slim

# Create a non-root user
RUN adduser --disabled-password --gecos "" appuser

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Switch to the non-root user
USER appuser

# Run the Streamlit app
CMD ["streamlit", "run", "src/visualization/streamlit_dashboard.py"]