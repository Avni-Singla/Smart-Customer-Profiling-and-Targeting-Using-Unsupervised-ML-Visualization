# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Dataset Loading
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

# Feature Engineering
df['Spending per Income'] = df['Spending Score (1-100)'] / df['Annual Income (k$)']
df['Age Group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 80], labels=['Young', 'Adult', 'Mid-Aged', 'Senior'])
df = pd.get_dummies(df, columns=['Gender', 'Age Group'], drop_first=True)

# Feature Selection and Scaling
features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Spending per Income'] + [col for col in df.columns if 'Gender_' in col or 'Age Group_' in col]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for KMeans')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['KMeans_Cluster'] = kmeans_labels

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=5)
agg_labels = agg.fit_predict(X_scaled)
df['Agglomerative_Cluster'] = agg_labels

# DBSCAN
db = DBSCAN(eps=1.2, min_samples=5)
db_labels = db.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = db_labels

gmm = GaussianMixture(n_components=5, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
df['GMM_Cluster'] = gmm_labels

# PCA Visualization
from sklearn.manifold import TSNE

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 5))
sns.scatterplot(data = df, x='PCA1', y='PCA2', hue='KMeans_Cluster', palette='tab10')
plt.title('KMeans Cluster Visualization (PCA)')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='GMM_Cluster', palette='Set1')
plt.title('GMM Cluster Visualization (PCA)')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='DBSCAN_Cluster', palette='Dark2')
plt.title('DBSCAN Cluster Visualization (PCA)')
plt.show()

# Segment Summary (example)
print(df.groupby('KMeans_Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Spending per Income']].mean())

# Predict Cluster with Classifier
X_train, X_test, y_train, y_test = train_test_split(X, kmeans_labels, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("Classifier Accuracy: {:.2f}%".format(clf.score(X_test, y_test)*100))

# Business Recommendations
segments = df.groupby('KMeans_Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
for i, row in segments.iterrows():
    print(f"Segment {i}: Income = {row[0]:.1f}, Spending Score = {row[1]:.1f}")
    if row[0] > 70 and row[1] > 70:
        print("→ High-income, high-spender. Target with luxury promotions.")
    elif row[0] < 40 and row[1] > 70:
        print("→ Low-income, high-spender. Offer discount and loyalty programs.")
    elif row[1] < 40:
        print("→ Low-spender. Consider engagement strategies or churn intervention.")
    else:
        print("→ Mid-range customer. Potential to grow with personalized offers.")
    print()
