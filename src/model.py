import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import save_pickle

def train_model(df, tfidf_path, similarity_path):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

    matrix = tfidf.fit_transform(df["combined"])
    similarity = cosine_similarity(matrix)

    # Save artifacts
    save_pickle(tfidf, tfidf_path)
    save_pickle(similarity, similarity_path)