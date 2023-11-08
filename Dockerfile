# Use an official Python runtime as a parent image, slim version for smaller size
FROM python:3.8-slim

# Set the working directory in the container to /root
WORKDIR /root

# Copy the current directory contents into the container at /root
COPY . /root

# Install any needed packages specified in your dependencies
RUN pip install --no-cache-dir torch==2.0.1 transformers pyext pytest modal

# Copy the required directories
COPY ./APPS /root/APPS
COPY ./Code-AI-Tree-Search /root/Code-AI-Tree-Search
COPY ./models /root/models

# Define environment variable
ENV TOKENIZERS_PARALLELISM=false

# # Run main.py when the container launches, passing the --remote argument
CMD ["python", "main.py", "--remote"]
