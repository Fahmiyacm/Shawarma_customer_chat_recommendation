# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Shawarma_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
