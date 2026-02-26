# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run the Streamlit application
# We use --server.address 0.0.0.0 to make it accessible outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]