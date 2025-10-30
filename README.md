# Cattle Milk Yield Prediction Pipeline

**Course:** CSCI461 - Introduction to Big Data  
**Assignment:** #1 - Fall 2025  
**Team Members:**
- Ebrahim Mohamed Elessawy
- Mohamed Al-tawapty
- Mahitab Ayman
- Mohammed Mahmood

## Project Overview
This project implements a complete data processing pipeline for cattle milk yield prediction using Docker containers. The pipeline processes raw data through multiple stages including ingestion, preprocessing, analytics, visualization, and clustering.

## Pipeline Workflow
customer-analytics/
ingest.py → data_raw.csv
preprocess.py → data_preprocessed.csv
analytics.py → insight1.txt, insight2.txt, insight3.txt
visualize.py → summary_plot.png
cluster.py → clusters.txt
summary.sh → All results copied to host



## Docker Commands

### Build Image
docker build -t cattle-analytics:latest .

text

### Run Container
  
**Windows CMD:** docker run -it --name cow-analytics -v %cd%:/app/pipeline cattle-analytics:latest

### Execute Pipeline
Inside container:
python ingest.py global_cattle_milk_yield_prediction_dataset.csv



### Extract Results
exit
bash summary.sh


## Implementation Details

### preprocess.py (5 Stages)
**Stage 1: Data Cleaning**
- Handles missing values (median for numeric, mode for categorical)
- Removes duplicate rows

**Stage 2: Feature Transformation**
- Encodes categorical variables using LabelEncoder
- Scales numeric features using StandardScaler

**Stage 3: Dimensionality Reduction**
- Applies PCA on scaled numeric features
- Automatically determines optimal components (5 components)

**Stage 4: Discretization** ⭐
- Bins Age_Months into 5 age categories
- Bins Milk_Yield_L into 4 productivity levels (quantile-based)
- Bins Body_Condition_Score into 4 health categories

**Stage 5: Feature Engineering**
- Creates 5 domain-specific derived features

### analytics.py
Generates three textual insights:
- Dataset Overview: Structure, dimensions, data types
- Feature Characteristics: Statistics, variability, correlations
- Distribution Patterns: Skewness, outliers, cluster analysis

### visualize.py
- Creates 8-panel visualization (summary_plot.png)
- Includes correlation heatmaps and distributions
- Handles various dataset sizes and types

### cluster.py
- Applies K-Means clustering (k=3)
- Clusters cattle by milk yield patterns
- Generates comprehensive cluster analysis report

### summary.sh
- Copies all results from Docker container to host
- Stops and removes the container
- Provides execution summary

## Sample Output
=== STAGE 4: DISCRETIZATION ===
✓ Created 'Age_Category' with 5 bins
Distribution: {'Adult (2-4yr)': 3456, 'Mature (4-6yr)': 2890, ...}
✓ Created 'Yield_Category' with 4 quantile-based bins
✓ Created 'BCS_Category' with 4 bins

Cluster Results:
Low Producers: 3421 samples - 15.3 L/day avg
Medium Producers: 3890 samples - 28.2 L/day avg
High Producers: 2674 samples - 42.7 L/day avg


## Docker Hub Deployment
docker pull lordadonis07/cattle-analytics:latest
docker run -it --name customer-analytics -v $(pwd):/app/pipeline lordadonis07/cattle-analytics:latest


