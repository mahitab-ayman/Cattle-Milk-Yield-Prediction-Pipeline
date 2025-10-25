import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    """
    Simplified clustering using only Milk_Yield_L feature
    """
    print("🔍 Applying K-Means clustering on Milk Yield...")
    
    # Load data
    data = pd.read_csv(sys.argv[1])
    print(f"📊 Data shape: {data.shape}")
    
    # Use only Milk_Yield_L feature
    if 'Milk_Yield_L' not in data.columns:
        print("❌ Error: Milk_Yield_L column not found")
        sys.exit(1)
    
    # Prepare data (single feature)
    X = data[['Milk_Yield_L']].fillna(data['Milk_Yield_L'].mean())
    
    # Check if we have enough data
    if len(X) < 3:
        print("❌ Not enough data for clustering")
        sys.exit(1)
    
    # Scale the single feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Generate cluster report
    report = "CATTLE MILK YIELD CLUSTERING\n"
    report += "=" * 40 + "\n\n"
    report += "CLUSTERING SETUP:\n"
    report += f"- Feature used: Milk_Yield_L\n"
    report += f"- Total cattle: {len(data):,}\n"
    report += f"- Clusters: 3\n\n"
    
    report += "PRODUCTION SEGMENTS:\n"
    report += "-" * 30 + "\n"
    
    # Calculate cluster statistics
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    milk_means = data.groupby(clusters)['Milk_Yield_L'].mean()
    
    segment_names = {
        0: "Low Producers",
        1: "Medium Producers", 
        2: "High Producers"
    }
    
    for cluster_id in range(3):
        count = cluster_counts.get(cluster_id, 0)
        percentage = (count / len(data)) * 100
        milk_avg = milk_means.get(cluster_id, 0)
        segment_name = segment_names.get(cluster_id, f"Segment {cluster_id}")
        
        report += f"{segment_name}:\n"
        report += f"  • Cattle count: {count:,} ({percentage:.1f}%)\n"
        report += f"  • Avg milk yield: {milk_avg:.1f} L/day\n"
        report += f"  • Yield range: {data[clusters == cluster_id]['Milk_Yield_L'].min():.1f}-{data[clusters == cluster_id]['Milk_Yield_L'].max():.1f} L\n\n"
    
    report += "MANAGEMENT RECOMMENDATIONS:\n"
    report += "-" * 30 + "\n"
    report += "• Low Producers: Consider health checks and nutrition review\n"
    report += "• Medium Producers: Maintain current management practices\n"
    report += "• High Producers: Focus on breeding and premium care\n"
    
    report += "\n" + "=" * 40 + "\n"
    report += "Clustering completed successfully!\n"
    
    # Save results
    with open("clusters.txt", "w") as f:
        f.write(report)
    
    print("✅ Clustering completed!")
    print(f"✅ {len(data):,} cattle segmented into 3 production tiers")
    print("✅ Results saved to clusters.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cluster.py <input_file>")
        sys.exit(1)
    main()