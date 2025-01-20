# app.py
import streamlit as st
import pickle
import os
import pandas as pd
from movieRecomendation import recommend_by_genre
from utils import load_data
from neural_cf import recommend_ncf
from collaborative_filtering import (
    prepare_data,
    train_nmf_model,
    predict_ratings,
    recommend_movies_nmf
)

# Load data
movies_df, ratings_df = load_data()

# Streamlit UI
st.title("Movie Recommendation System")

# Sidebar for input preferences
st.sidebar.header("Input Preferences")
recommendation_method = st.sidebar.radio(
    "Select Recommendation Method:",
    ["Genre-Based", "Collaborative Filtering", "Neural Collaborative Filtering"]
)

# Mandatory Genre Selection for Genre-Based Recommendations
favorite_genres = st.sidebar.multiselect(
    "Select your favorite genres (required):",
    ["Action", "Drama", "Comedy", "Horror", "Sci-Fi", "Romance", "Adventure"]
)
if not favorite_genres and recommendation_method == "Genre-Based":
    st.warning("You must select at least one genre.")

# Recommendation Options
num_recommendations = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

if recommendation_method == "Genre-Based":
    if st.sidebar.button("Get Recommendations"):
        try:
            recommendations = recommend_by_genre(favorite_genres, num_recommendations)
            if recommendations:
                st.header("Your Recommendations:")
                for movie in recommendations:
                    st.write(f"- {movie['title']} (ID: {movie['movie_id']})")
            else:
                st.warning("No recommendations found for the selected genres.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif recommendation_method == "Collaborative Filtering":
    user_id = st.sidebar.number_input("Enter User ID:", min_value=1, max_value=943)
    if st.sidebar.button("Get ML Recommendations"):
        try:
            if not os.path.exists("nmf_model.pkl"):
                st.warning("Training the NMF model. This may take a while...")
                user_item_matrix = prepare_data(ratings_df)
                user_factors, movie_factors, _ = train_nmf_model(user_item_matrix, n_components=50, alpha=0.1)
                reconstructed_matrix = predict_ratings(user_factors, movie_factors)
                with open("nmf_model.pkl", "wb") as f:
                    pickle.dump((user_factors, movie_factors), f)
            else:
                with open("nmf_model.pkl", "rb") as f:
                    user_factors, movie_factors = pickle.load(f)
                reconstructed_matrix = predict_ratings(user_factors, movie_factors)
                user_item_matrix = prepare_data(ratings_df)

            recommendations = recommend_movies_nmf(
                user_id=user_id,
                user_item_matrix=user_item_matrix,
                reconstructed_matrix=reconstructed_matrix,
                movies_df=movies_df,
                ratings_df=ratings_df,
                top_n=num_recommendations
            )

            if recommendations.empty:
                st.warning("No recommendations found for this user.")
            else:
                st.header("Your ML-Based Recommendations:")
                for _, movie in recommendations.iterrows():
                    st.write(f"- {movie['title']} (ID: {movie['movie_id']})")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif recommendation_method == "Neural Collaborative Filtering":
    user_id = st.sidebar.number_input("Enter User ID:", min_value=1, max_value=943)
    if st.sidebar.button("Get NCF Recommendations"):
        try:
            recommendations = recommend_ncf(user_id, top_n=num_recommendations)
            if recommendations:
                st.header("Your NCF-Based Recommendations:")
                for movie in recommendations:
                    st.write(f"- {movie['title']} (ID: {movie['movie_id']})")
            else:
                st.warning("No recommendations found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
