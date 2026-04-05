import pickle
import os

def load_pickle(path):
    return pickle.load(open(path, "rb"))

def save_pickle(obj, path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(obj, open(path, "wb"))

def load_model(tfidf_path, similarity_path):
    tfidf = load_pickle(tfidf_path)
    similarity = load_pickle(similarity_path)
    return tfidf, similarity