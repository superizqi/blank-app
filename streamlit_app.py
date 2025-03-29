import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title
st.title("Customer Segmentation with Clustering")
st.write("This app clusters customers based on purchasing behavior using K-Means.")

# Load Dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url)
    return df

df = load_data()

# Data Cleaning
df = df.dropna()
df = df[df['Quantity'] > 0]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Feature Engineering (RFM Analysis)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

# Normalize Data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Sidebar: Select number of clusters
n_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=4, step=1)

# Apply K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Display Clustered Data
st.write("### Clustered Data Sample", rfm.head())

# Visualization
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Cluster'], palette='viridis', ax=ax)
plt.xlabel("Recency (days since last purchase)")
plt.ylabel("Monetary Value (Total Spend)")
plt.title("Customer Clusters")
st.pyplot(fig)

st.write("Move the slider to change the number of clusters dynamically.")
