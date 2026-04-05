import streamlit as st

def movie_card(row):
    st.markdown(f"""
    ### � {row['title']}
    ⭐ Rating: {row['vote_average']}  
    🔥 Popularity: {round(row['popularity'], 2)}
    """)