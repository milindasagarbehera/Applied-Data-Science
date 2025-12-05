# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Copy the requirements file and install dependencies
# This is done first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files, including the model and tokenizer
# This includes: app.py, spam_lstm_model.h5, and tokenizer.pickle
COPY . .

# Command to run the Streamlit application
# The --server.port and --server.address flags ensure it binds correctly inside the container
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]