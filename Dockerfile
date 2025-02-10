FROM python:3.9-slim

# Create a non-root user
RUN adduser --disabled-password --gecos "" appuser

# Set working directory
WORKDIR /app

# Install dependencies as root
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Switch to the non-root user
USER appuser

# Copy project files
COPY . .

# Run the Streamlit app
CMD ["streamlit", "run", "src/visualization/streamlit_dashboard.py"]