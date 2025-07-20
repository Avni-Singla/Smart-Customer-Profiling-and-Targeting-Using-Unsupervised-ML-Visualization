# Advanced Customer Segmentation Streamlit Dashboard with Personas, Prediction, and Metrics

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("Customer Segmentation Dashboard")

# Load Data
df = pd.read_csv("C:/Users/avnis/PycharmProjects/pythonProject1/Customer Segmentation Project/Mall_Customers.csv")
df['Spending per Income'] = df['Spending Score (1-100)'] / df['Annual Income (k$)']
df['Age Group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 80], labels=['Young', 'Adult', 'Mid-Aged', 'Senior'])
df = pd.get_dummies(df, columns=['Gender', 'Age Group'], drop_first=True)

# Select Features
features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Spending per Income'] + [col for col in df.columns if 'Gender_' in col or 'Age Group_' in col]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar Configuration
st.sidebar.header("Clustering Settings")
algorithm = st.sidebar.selectbox("Select Clustering Algorithm", ["KMeans", "Agglomerative", "DBSCAN", "GMM"])
num_clusters = st.sidebar.slider("Number of Clusters (for KMeans, Agglomerative, GMM)", 2, 10, 5)
eps = st.sidebar.slider("DBSCAN eps (only used for DBSCAN)", 0.1, 5.0, 1.2, step=0.1)

# Clustering
def get_cluster_labels(algorithm, X_scaled, n_clusters=5, eps=1.2):
    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == "GMM":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        return model.fit_predict(X_scaled)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=5)
    return model.fit_predict(X_scaled)

cluster_labels = get_cluster_labels(algorithm, X_scaled, num_clusters, eps)
df['Cluster'] = cluster_labels

# Clustering Metric: Silhouette Score
if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
    score = silhouette_score(X_scaled, cluster_labels)
    st.metric(label="Silhouette Score", value=f"{score:.2f}")

# Dimensionality Reduction for Plot
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Visualize Clusters
st.subheader("Cluster Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax)
plt.title(f"{algorithm} Clustering Results (PCA-reduced)")
st.pyplot(fig)

# Cluster Averages
st.subheader("Cluster Summary")
avg_metrics = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Spending per Income']].mean()
st.dataframe(avg_metrics.style.format("{:.2f}"))

# Business Personas
st.subheader("Segment Personas")
personas = {
    0: "Budget-Conscious Shoppers",
    1: "Mid-Income Moderate Spenders",
    2: "Value Seekers",
    3: "Affluent Enthusiasts",
    4: "Passive Participants"
}
for cluster_id, description in personas.items():
    if cluster_id in df['Cluster'].unique():
        st.markdown(f"**Cluster {cluster_id}:** {description}")

# Business Insights
st.subheader("Business Recommendations")
for i, row in avg_metrics.iterrows():
    st.markdown(f"**Segment {i}:** Income = {row[0]:.1f}, Spending Score = {row[1]:.1f}")
    if row[0] > 70 and row[1] > 70:
        st.success("High-income, high-spender → Target with luxury promotions.")
    elif row[0] < 40 and row[1] > 70:
        st.warning("Low-income, high-spender → Offer discounts and loyalty programs.")
    elif row[1] < 40:
        st.info("Low spender → Consider engagement strategies or churn intervention.")
    else:
        st.info("Mid-range customer → Potential to grow with personalized offers.")

# Compare clustering algorithms by Silhouette Score
st.subheader("Algorithm Comparison (Silhouette Scores)")
def compare_algorithms(X_scaled, n_clusters):
    scores = {}
    for algo in ["KMeans", "Agglomerative", "GMM"]:
        try:
            labels = get_cluster_labels(algo, X_scaled, n_clusters)
            if len(set(labels)) > 1 and -1 not in labels:
                score = silhouette_score(X_scaled, labels)
                scores[algo] = score
        except:
            scores[algo] = None
    return scores

algo_scores = compare_algorithms(X_scaled, num_clusters)
if algo_scores:
    fig_score, ax_score = plt.subplots()
    ax_score.bar(algo_scores.keys(), algo_scores.values(), color='skyblue')
    ax_score.set_ylabel("Silhouette Score")
    ax_score.set_title("Algorithm Performance Comparison")
    st.pyplot(fig_score)

# Train classifier to predict clusters (all except DBSCAN)
if algorithm != "DBSCAN":
    st.subheader("Predict Segment for New Customer")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, cluster_labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)

    st.metric("Classifier Accuracy", f"{accuracy * 100:.2f}%")

    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # Input fields
    income = st.number_input("Annual Income (k$)", min_value=10, max_value=150, value=60)
    score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=18, max_value=80, value=30)

    # Derived inputs
    spi = score / income
    age_group = pd.cut([age], bins=[18, 30, 45, 60, 80], labels=['Young', 'Adult', 'Mid-Aged', 'Senior'])[0]

    input_data = {
        'Annual Income (k$)': income,
        'Spending Score (1-100)': score,
        'Spending per Income': spi,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Age Group_Adult': 1 if age_group == 'Adult' else 0,
        'Age Group_Mid-Aged': 1 if age_group == 'Mid-Aged' else 0,
        'Age Group_Senior': 1 if age_group == 'Senior' else 0
    }
    for col in features:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df[features])
    predicted_cluster = clf.predict(input_scaled)[0]

    st.success(f"Predicted Segment: Cluster {predicted_cluster} — {personas.get(predicted_cluster, 'Unknown')}")

# Optionally Show Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Run StreaLit App
# streamlit run dashboard.py