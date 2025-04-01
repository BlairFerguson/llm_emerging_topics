# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary application files (adjust as needed)
COPY api_rag_spanish.py .

# Set environment variable to disable GPU use
ENV USE_CPU=True

# Command to run the FastAPI app
CMD ["uvicorn", "api_rag_spanish:app", "--host", "0.0.0.0", "--port", "8000"]
