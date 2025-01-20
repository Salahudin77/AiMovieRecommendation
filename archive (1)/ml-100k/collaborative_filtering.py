# collaborative_filtering.py
from sklearn.decomposition import NMF
from surprise import SVD, Dataset, Reader, accuracy
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils import load_data  # Use centralized data loading

# Load data
movies_df, ratings_df = load_data()

def prepare_data(ratings_df):
    """
    Normalize ratings to the range [0, 1] for compatibility with NMF.
    """
    if ratings_df.empty:
        raise ValueError("Ratings DataFrame is empty!")
    ratings_df['rating'] = (ratings_df['rating'] - ratings_df['rating'].min()) / (ratings_df['rating'].max() - ratings_df['rating'].min())
    user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    return user_item_matrix

# Step 2: Train NMF Model
def train_nmf_model(user_item_matrix, n_components=50, alpha=0.1):
     """
    Train an NMF model using the given user-item matrix.
    Args:
        user_item_matrix: A pandas DataFrame of user-item interactions.
    Returns:
        Trained NMF model components (user and item factors).

    """
    nmf = NMF(n_components=n_components, max_iter=500, random_state=42, alpha_W=alpha, alpha_H=alpha)
    user_factors = nmf.fit_transform(user_item_matrix)
    movie_factors = nmf.components_
    with open("nmf_model.pkl", "wb") as f:
        pickle.dump((user_factors, movie_factors), f)
    print(f"NMF model trained. User factors shape: {user_factors.shape}, Movie factors shape: {movie_factors.shape}")
    return user_factors, movie_factors, nmf

# Step 3: Train SVD Model
def train_svd_model(ratings_df):
    """
    Train SVD using the surprise library.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()

    algo = SVD(n_factors=50, reg_all=0.1)
    algo.fit(trainset)

    with open("svd_model.pkl", "wb") as f:
        pickle.dump(algo, f)

    print("SVD model trained and saved.")
    return algo

# Step 4: Predict Ratings
def predict_ratings(user_factors, movie_factors):
    """
    Reconstruct the user-item matrix from NMF factors.
    """
    reconstructed_matrix = np.dot(user_factors, movie_factors)
    print(f"Reconstructed matrix shape: {reconstructed_matrix.shape}")
    return reconstructed_matrix

def recommend_movies_nmf(user_id, user_item_matrix, reconstructed_matrix, movies_df, ratings_df, top_n=5, favorite_genres=None):
    """
    Generate recommendations using NMF, with optional genre filtering.
    """
    # Ensure user_item_matrix and reconstructed_matrix are not None
    if user_item_matrix is None or reconstructed_matrix is None:
        raise ValueError("User-item matrix and reconstructed matrix cannot be None.")

    # Get the user's row in the reconstructed matrix
    user_index = user_id - 1
    if user_index < 0 or user_index >= reconstructed_matrix.shape[0]:
        # User not in the dataset, recommend popular movies instead
        return recommend_popular_movies(ratings_df, movies_df, top_n)

    # Calculate user-specific recommendations
    user_ratings = reconstructed_matrix[user_index]
    rated_movies = user_item_matrix.iloc[user_index] > 0
    user_ratings[rated_movies] = 0  # Ignore already-rated movies
    top_indices = np.argsort(user_ratings)[-top_n:][::-1]  # Get top N movies

    recommendations = movies_df[movies_df['movie_id'].isin(top_indices)]

    # Filter recommendations by genre if selected
    if favorite_genres:
        # Dynamically find columns for the selected genres
        available_genres = [col.replace("genre_", "") for col in movies_df.columns if col.startswith("genre_")]
        selected_genres = [genre for genre in favorite_genres if genre in available_genres]

        if selected_genres:
            genre_columns = [f"genre_{genre}" for genre in selected_genres]
            recommendations = recommendations[
                recommendations[genre_columns].any(axis=1)
            ]

    return recommendations
