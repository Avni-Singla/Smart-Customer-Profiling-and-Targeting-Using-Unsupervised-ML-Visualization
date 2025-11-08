# dashboard.py
# Streamlit Customer Segmentation Dashboard (Plotly-based, Pillow-free)
import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="PersonaVision — Customer Segmentation", layout="wide")
st.title("PersonaVision — Smart Customer Segmentation & Profiling")

# ---------------------------
# Helper: Load and preprocess
# ---------------------------
REQUIRED_COLUMNS = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']

def read_csv_from_upload(uploaded_file):
    try:
        bytes_data = uploaded_file.read()
        return pd.read_csv(io.BytesIO(bytes_data))
    except Exception as e:
        st.error(f"Unable to read uploaded file: {e}")
        return None

def load_and_prepare_dataframe(uploaded_file):
    """
    Load CSV either from uploaded file or from repo root Mall_Customers.csv.
    Preprocess: create Spending per Income, Age Group, one-hot encoding for Gender and Age Group.
    Returns processed df, features list, scaler instance, and scaled X.
    """
    # Load
    df = None
    if uploaded_file is not None:
        df = read_csv_from_upload(uploaded_file)
    else:
        local_path = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
        else:
            st.error("No dataset found. Upload a CSV file or place 'Mall_Customers.csv' in the app folder.")
            return None, None, None, None

    # Basic validation
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Dataset is missing required column(s): {missing}. Expected columns: {REQUIRED_COLUMNS}")
        return None, None, None, None

    # Copy to avoid modifying original
    df = df.copy()

    # Create derived features safely
    df['Annual Income (k$)'] = pd.to_numeric(df['Annual Income (k$)'], errors='coerce')
    df['Spending Score (1-100)'] = pd.to_numeric(df['Spending Score (1-100)'], errors='coerce')
    df['Spending per Income'] = df.apply(
        lambda row: (row['Spending Score (1-100)'] / row['Annual Income (k$)'])
        if pd.notnull(row['Annual Income (k$)']) and row['Annual Income (k$)'] != 0 else 0,
        axis=1
    )

    # Age group bucketing
    try:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Age Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 120], labels=['Young', 'Adult', 'Mid-Aged', 'Senior'])
    except Exception:
        df['Age Group'] = 'Unknown'

    # One-hot encoding for Gender and Age Group (drop_first to avoid collinearity)
    df = pd.get_dummies(df, columns=['Gender', 'Age Group'], drop_first=True, dummy_na=False)

    # Select features for clustering (keep order consistent)
    feature_candidates = ['Annual Income (k$)', 'Spending Score (1-100)', 'Spending per Income']
    encoded_cols = [col for col in df.columns if col.startswith('Gender_') or col.startswith('Age Group_')]
    features = feature_candidates + encoded_cols

    # Ensure features exist
    features = [f for f in features if f in df.columns]

    if len(features) < 3:
        st.error("Not enough features available for clustering after preprocessing.")
        return None, None, None, None

    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, features, scaler, X_scaled

# ---------------------------
# UI: Data Upload / Load
# ---------------------------
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload Mall_Customers CSV", type=["csv"])
st.sidebar.markdown("If no file is uploaded, the app will attempt to read `Mall_Customers.csv` from the app folder.")

df, features, scaler, X_scaled = load_and_prepare_dataframe(uploaded_file)

# If loading failed, stop further execution
if df is None:
    st.stop()

# ---------------------------
# Sidebar Configuration - Clustering
# ---------------------------
st.sidebar.header("Clustering Settings")
algorithm = st.sidebar.selectbox("Select Clustering Algorithm", ["KMeans", "Agglomerative", "DBSCAN", "GMM"])
num_clusters = st.sidebar.slider("Number of Clusters (KMeans, Agglomerative, GMM)", 2, 10, 5)
eps = st.sidebar.slider("DBSCAN eps (only used for DBSCAN)", 0.1, 5.0, 1.2, step=0.1)
min_samples = st.sidebar.slider("DBSCAN min_samples", 3, 10, 5)

# ---------------------------
# Clustering function
# ---------------------------
def get_cluster_labels(algorithm, X_scaled, n_clusters=5, eps=1.2, min_samples=5):
    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = model.fit_predict(X_scaled)
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "GMM":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    else:
        labels = np.zeros(X_scaled.shape[0], dtype=int)
    return labels

cluster_labels = get_cluster_labels(algorithm, X_scaled, num_clusters, eps, min_samples)
df['Cluster'] = cluster_labels

# ---------------------------
# Metrics: Silhouette (if valid)
# ---------------------------
def valid_silhouette(labels):
    unique_labels = [lab for lab in set(labels) if lab != -1]
    if len(unique_labels) < 2:
        return False
    sizes = [np.sum(labels == lab) for lab in unique_labels]
    if min(sizes) < 2:
        return False
    return True

if valid_silhouette(cluster_labels):
    try:
        score = silhouette_score(X_scaled, cluster_labels)
        st.metric(label="Silhouette Score", value=f"{score:.2f}")
    except Exception:
        pass

# ---------------------------
# Dimensionality Reduction for Plot (PCA)
# ---------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# ---------------------------
# Visualization: Clusters (Plotly)
# ---------------------------
st.subheader("Cluster Visualization (PCA projection)")
# Convert cluster to string for color consistency
df['_Cluster_str'] = df['Cluster'].astype(str)
fig_scatter = px.scatter(
    df,
    x='PCA1',
    y='PCA2',
    color='_Cluster_str',
    labels={'_Cluster_str': 'Cluster'},
    hover_data=[col for col in df.columns if col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']],
    title=f"{algorithm} Clustering Results (PCA-reduced)"
)
fig_scatter.update_traces(marker=dict(size=8))
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------
# Cluster Summary
# ---------------------------
st.subheader("Cluster Summary")
summary_cols = [c for c in ['Annual Income (k$)', 'Spending Score (1-100)', 'Spending per Income'] if c in df.columns]
if summary_cols:
    avg_metrics = df.groupby('Cluster')[summary_cols].mean()
    st.dataframe(avg_metrics.style.format("{:.2f}"))
else:
    st.info("Not enough columns available for cluster summaries.")

# ---------------------------
# Personas & Business Insights
# ---------------------------
st.subheader("Segment Personas")
default_personas = {
    0: "Budget-Conscious Shoppers",
    1: "Mid-Income Moderate Spenders",
    2: "Value Seekers",
    3: "Affluent Enthusiasts",
    4: "Passive Participants"
}
for cluster_id, description in default_personas.items():
    if cluster_id in df['Cluster'].unique():
        st.markdown(f"**Cluster {cluster_id}:** {description}")

st.subheader("Business Recommendations")
if 'Cluster' in df.columns and summary_cols:
    for i, row in avg_metrics.iterrows():
        inc = float(row[summary_cols[0]]) if len(summary_cols) > 0 else 0.0
        score = float(row[summary_cols[1]]) if len(summary_cols) > 1 else 0.0
        st.markdown(f"**Segment {i}:** Income = {inc:.1f}, Spending Score = {score:.1f}")
        if inc > 70 and score > 70:
            st.success("High-income, high-spender → Target with luxury promotions.")
        elif inc < 40 and score > 70:
            st.warning("Low-income, high-spender → Offer discounts and loyalty programs.")
        elif score < 40:
            st.info("Low spender → Consider engagement strategies or churn intervention.")
        else:
            st.info("Mid-range customer → Potential to grow with personalized offers.")
else:
    st.info("Business recommendations require numeric columns (income & spending score).")

# ---------------------------
# Algorithm Comparison by Silhouette (Plotly Bar)
# ---------------------------
st.subheader("Algorithm Comparison (Silhouette Scores)")
def compare_algorithms(X_scaled, n_clusters):
    scores = {}
    for algo in ["KMeans", "Agglomerative", "GMM"]:
        try:
            labels = get_cluster_labels(algo, X_scaled, n_clusters)
            if valid_silhouette(labels):
                scores[algo] = float(silhouette_score(X_scaled, labels))
            else:
                scores[algo] = None
        except Exception:
            scores[algo] = None
    return scores

algo_scores = compare_algorithms(X_scaled, num_clusters)
if any(v is not None for v in algo_scores.values()):
    keys = list(algo_scores.keys())
    vals = [v if v is not None else 0 for v in algo_scores.values()]
    fig_bar = px.bar(x=keys, y=vals, labels={'x':'Algorithm','y':'Silhouette Score'},
                     title='Algorithm Performance Comparison')
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Not enough valid clusters for algorithm comparison.")

# ---------------------------
# Classifier: Train to predict clusters (exclude DBSCAN noise)
# ---------------------------
if algorithm != "DBSCAN" and len(set(cluster_labels)) > 1:
    st.subheader("Predict Segment for New Customer")
    valid_idx = np.where(cluster_labels != -1)[0]
    if len(valid_idx) >= 2:
        X_valid = X_scaled[valid_idx]
        y_valid = cluster_labels[valid_idx]
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            st.metric("Classifier Accuracy", f"{accuracy * 100:.2f}%")

            # Confusion matrix (Plotly)
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f"Pred_{c}" for c in np.unique(y_test)],
                y=[f"True_{c}" for c in np.unique(y_test)],
                colorscale='Blues',
                showscale=True,
                text=cm,
                texttemplate="%{text}"
            ))
            fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
            st.plotly_chart(fig_cm, use_container_width=True)

            # Input fields for single prediction
            st.markdown("**Predict segment for a new single customer**")
            income = st.number_input("Annual Income (k$)", min_value=1.0, max_value=100000.0, value=60.0, step=1.0)
            score = st.number_input("Spending Score (1-100)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

            spi = (score / income) if income != 0 else 0
            age_group_label = pd.cut([age], bins=[0, 30, 45, 60, 120], labels=['Young', 'Adult', 'Mid-Aged', 'Senior'])[0]

            # Construct input aligned to features
            input_data = {}
            for f in features:
                if f == 'Annual Income (k$)':
                    input_data[f] = income
                elif f == 'Spending Score (1-100)':
                    input_data[f] = score
                elif f == 'Spending per Income':
                    input_data[f] = spi
                elif f.startswith('Gender_'):
                    input_data[f] = 1 if f == f"Gender_{gender}" else 0
                elif f.startswith('Age Group_'):
                    col_label = f.replace('Age Group_', '')
                    input_data[f] = 1 if str(age_group_label) == col_label else 0
                else:
                    input_data[f] = 0

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df[features])
            predicted_cluster = clf.predict(input_scaled)[0]
            st.success(f"Predicted Segment: Cluster {predicted_cluster} — {default_personas.get(predicted_cluster, 'Unknown')}")
        except Exception as e:
            st.error(f"Unable to train classifier or predict: {e}")
    else:
        st.info("Not enough valid labeled samples to train a classifier (after excluding noise).")

# ---------------------------
# Optionally Show Raw Data
# ---------------------------
if st.checkbox("Show Raw Data"):
    st.dataframe(df.reset_index(drop=True))
