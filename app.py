import streamlit as st
import sys
import os
sys.path.append('.')

from config import DATA_PATH, SIMILARITY_PATH, TFIDF_PATH
from src.preprocess import load_data, create_features
from src.utils import load_pickle
from src.recommender import get_recommendations
from src.model import train_model
from app.ui import render_ui

st.set_page_config(layout="wide")

def load_data_safe():
    """Load data with error handling"""
    try:
        return load_data(DATA_PATH)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

def generate_model():
    """Generate model if not present"""
    st.warning("⏳ First-time setup: Generating recommendation model...")
    with st.spinner("This may take 2-3 minutes on first run..."):
        try:
            df = load_data_safe()
            df_featured = create_features(df)
            train_model(df_featured, TFIDF_PATH, SIMILARITY_PATH)
            st.success("✅ Model ready!")
        except Exception as e:
            st.error(f"❌ Error generating model: {e}")
            st.stop()

# Load data
df = load_data_safe()

# Load or generate model
if not (os.path.exists(SIMILARITY_PATH)):
    generate_model()

try:
    similarity = load_pickle(SIMILARITY_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Render UI
render_ui(df, get_recommendations, similarity)