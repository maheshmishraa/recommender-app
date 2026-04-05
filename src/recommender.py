from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["combined"])

    similarity = cosine_similarity(matrix)
    return similarity

def get_recommendations(title, df, similarity, top_n=5):
    if title not in df["title"].values:
        return ["Movie not found"]

    idx = df[df["title"] == title].index[0]

    sim_scores = list(enumerate(similarity[idx]))

    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:50]

    movie_indices = [i[0] for i in sim_scores]

    # Create scoring dataframe
    candidates = df.iloc[movie_indices].copy()
    candidates["similarity_score"] = [i[1] for i in sim_scores]

    # Hybrid score
    candidates["final_score"] = (
        0.6 * candidates["similarity_score"] +
        0.2 * (candidates["vote_average"] / 10) +
        0.2 * (candidates["popularity"] / candidates["popularity"].max())
    )

    # Sort final results
    candidates = candidates.sort_values("final_score", ascending=False)

    return candidates[["title", "vote_average", "popularity"]].head(top_n)