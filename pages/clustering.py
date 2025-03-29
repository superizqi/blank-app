import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import sweetviz as sv
import tempfile
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title
st.title("Customer Segmentation & Exploratory Data Analysis")
st.write("This app clusters customers based on purchasing behavior using K-Means and provides an EDA report.")

# Load Dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
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

# Scatter Plot for Clustering
fig = px.scatter(rfm, x="Recency", y="Monetary", color=rfm["Cluster"].astype(str), 
                 title="Customer Clusters",
                 labels={"Cluster": "Customer Segment"},
                 hover_data=rfm.columns)

st.plotly_chart(fig)

# Additional EDA Plots
st.write("### Distribution of RFM Features")

# Histogram
fig_hist = px.histogram(rfm, x=["Recency", "Frequency", "Monetary"], title="Histogram of RFM Features", barmode="overlay")
st.plotly_chart(fig_hist)

# Boxplot
fig_box = px.box(rfm, x="Cluster", y="Monetary", color="Cluster",
                 title="Boxplot of Monetary Value per Cluster")
st.plotly_chart(fig_box)

# Generate EDA Report using Sweetviz
st.write("### Automated Exploratory Data Analysis Report")
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
    report = sv.analyze(rfm)
    report.show_html(tmp_file.name)
    st.download_button(label="Download EDA Report", data=open(tmp_file.name, "rb").read(), file_name="eda_report.html", mime="text/html")
