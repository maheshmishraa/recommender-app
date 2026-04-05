import streamlit as st
from .components import movie_card

def render_ui(df, recommend_fn, similarity):
    st.title("🎬 Smart Movie Recommendation System")

    selected_movie = st.selectbox(
        "Select a movie you like:",
        sorted(df["title"].unique())
    )

    if st.button("Recommend"):
        results = recommend_fn(selected_movie, df, similarity)

        if isinstance(results, list):
            st.error(results[0])
            return

        st.subheader("Top Recommendations:")

        for _, row in results.iterrows():
            movie_card(row)