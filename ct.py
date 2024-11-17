# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns

# Load production data
# Replace 'production_data.csv' with the path to your data file
data = pd.read_csv('production_data.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Step 1: Data Preprocessing
# Check for missing values
if data.isnull().sum().any():
    print("Handling missing values...")
    data.fillna(method='ffill', inplace=True)

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['cycle_time', 'defect_rate', 'utilization', 'downtime']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Step 2: Identify Bottlenecks using Clustering
# Use KMeans clustering to identify bottlenecks
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(data[numerical_cols])

# Visualize bottlenecks
sns.pairplot(data, hue='cluster', vars=numerical_cols)
plt.title("Clustering Results: Identifying Bottlenecks")
plt.show()

# Step 3: Optimize Workflows using Linear Programming
# Example workflow optimization problem:
# Minimize cost while meeting production demand
# Define cost coefficients for tasks
costs = np.array([5, 7, 9])  # Example costs per unit task
constraints = [
    [1, 1, 1],  # Task contributions to demand
]
bounds = [(0, None), (0, None), (0, None)]  # No negative production
demand = [50]  # Demand constraint

# Solve the optimization problem
result = linprog(c=costs, A_eq=constraints, b_eq=demand, bounds=bounds, method='highs')

# Display optimization results
print("\nOptimization Results:")
if result.success:
    print(f"Optimal Allocation: {result.x}")
    print(f"Minimum Cost: {result.fun}")
else:
    print("Optimization failed.")

# Step 4: Generate Insights
# Summary of clusters
cluster_summary = data.groupby('cluster').mean()
print("\nCluster Summary:")
print(cluster_summary)

# Recommendations based on insights
print("\nRecommendations:")
print("1. Address bottlenecks in clusters with high defect rates or downtime.")
print("2. Optimize resource allocation based on workflow efficiency.")

# Save processed data and results for reporting
data.to_csv('optimized_production_data.csv', index=False)
print("\nProcessed data saved to 'optimized_production_data.csv'.")