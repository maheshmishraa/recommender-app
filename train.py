from src.preprocess import load_data, create_features
from src.model import train_model
from config import DATA_PATH, TFIDF_PATH, SIMILARITY_PATH

df = load_data(DATA_PATH)
df = create_features(df)

train_model(df, TFIDF_PATH, SIMILARITY_PATH)

print("Model trained and saved!")