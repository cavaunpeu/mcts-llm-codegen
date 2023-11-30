# Use an official Python runtime as a parent image, slim version for smaller size
FROM python:3.8-slim

# Install system dependencies required for compiling certain Python packages like psutil
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev git && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from your host to your current location
COPY ./requirements.txt /root

# Install any needed packages specified in your dependencies
RUN pip install --no-cache-dir -r /root/requirements.txt

# Copy the current directory contents into the container at /root
COPY . /root

# Set the working directory in the container to /root
WORKDIR /root
