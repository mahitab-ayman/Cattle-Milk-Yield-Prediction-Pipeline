# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set maintainer information
LABEL maintainer="customer-analytics-team"
LABEL description="Docker container for cattle milk yield prediction pipeline"

# Install required Python packages
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    requests

# Create working directory inside the container
RUN mkdir -p /app/pipeline/

# Set working directory
WORKDIR /app/pipeline/

# Copy all project scripts into the container
COPY ingest.py /app/pipeline/
COPY preprocess.py /app/pipeline/
COPY analytics.py /app/pipeline/
COPY visualize.py /app/pipeline/
COPY cluster.py /app/pipeline/

# Start an interactive bash shell when container runs
CMD ["/bin/bash"]
