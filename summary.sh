#!/bin/bash


# Container name
CONTAINER_NAME="customer-analytics"


# Create results directory on host if it doesn't exist
mkdir -p customer-analytics/results/


# Copy all generated outputs from container to host
echo "📦 Copying outputs from container to host..."
docker cp ${CONTAINER_NAME}:/app/pipeline/data_raw.csv ./customer-analytics/results/
docker cp ${CONTAINER_NAME}:/app/pipeline/data_preprocessed.csv ./customer-analytics/results/
docker cp ${CONTAINER_NAME}:/app/pipeline/insight1.txt ./customer-analytics/results/
docker cp ${CONTAINER_NAME}:/app/pipeline/insight2.txt ./customer-analytics/results/
docker cp ${CONTAINER_NAME}:/app/pipeline/insight3.txt ./customer-analytics/results/
docker cp ${CONTAINER_NAME}:/app/pipeline/clusters.txt ./customer-analytics/results/
docker cp ${CONTAINER_NAME}:/app/pipeline/summary_plot.png ./customer-analytics/results/

echo "✅ All outputs copied successfully to customer-analytics/results/"

# Stop and remove the container
echo "🛑 Stopping container..."
docker stop ${CONTAINER_NAME}

echo "🗑️ Removing container..."
docker rm ${CONTAINER_NAME}

echo "✨ Pipeline completed successfully!"
