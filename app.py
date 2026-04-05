import streamlit as st
import sys
sys.path.append('.')

from config import DATA_PATH, SIMILARITY_PATH
from src.preprocess import load_data
from src.utils import load_pickle
from src.recommender import get_recommendations
from app.ui import render_ui

st.set_page_config(layout="wide")

@st.cache_data
def load_all():
    df = load_data(DATA_PATH)
    similarity = load_pickle(SIMILARITY_PATH)
    return df, similarity

df, similarity = load_all()

render_ui(df, get_recommendations, similarity)