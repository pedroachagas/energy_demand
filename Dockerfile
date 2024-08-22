# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Setup
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Run modeling_pipeline.py when the container launches
CMD ["python", "-m", "src.models.modeling_pipeline"]