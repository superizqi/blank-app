import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download stopwords
nltk.download("stopwords")
nltk.download("punkt")

# Title
st.title("Topic Modeling with Latent Dirichlet Allocation (LDA)")
st.write("This app extracts topics from customer reviews using LDA.")

# Load Dataset
@st.cache_data
def load_data():
    url = "https://www.kaggleusercontent.com/datasets/snap/amazon-fine-food-reviews/amazon-fine-food-reviews.zip"
    df = pd.read_csv(url, compression='zip', usecols=['Text'])
    return df

df = load_data()
st.write("### Sample Reviews", df.sample(5))

# Preprocessing Function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())  # Tokenization & Lowercase
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove Stopwords & Punctuation
    return tokens

df["processed"] = df["Text"].apply(preprocess_text)

# Create Dictionary & Corpus
dictionary = corpora.Dictionary(df["processed"])
corpus = [dictionary.doc2bow(text) for text in df["processed"]]

# Sidebar: Select Number of Topics
n_topics = st.sidebar.slider("Select Number of Topics", min_value=2, max_value=10, value=4, step=1)

# Train LDA Model
lda_model = models.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=10)

# Display Topics
st.write("### Extracted Topics")
topics = lda_model.print_topics(num_words=5)
for i, topic in topics:
    st.write(f"**Topic {i+1}:** {topic}")

# WordCloud for Topics
fig, axes = plt.subplots(nrows=1, ncols=n_topics, figsize=(15, 5))
for i, topic in enumerate(topics):
    topic_words = dict(lda_model.show_topic(i, 10))
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(topic_words)
    axes[i].imshow(wordcloud, interpolation="bilinear")
    axes[i].axis("off")
    axes[i].set_title(f"Topic {i+1}")

st.pyplot(fig)
