# utils.py
import os
import pandas as pd

def load_data():
    """
    Load movies and ratings data.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_dir, "u.item")
    ratings_path = os.path.join(base_dir, "u.data")

    print(f"Movies file path: {movies_path}")
    print(f"Ratings file path: {ratings_path}")

    movies_df = pd.read_csv(movies_path, sep='|', encoding='latin1', header=None)
    ratings_df = pd.read_csv(ratings_path, sep='\t', header=None)

    # Assign column names
    movies_df.columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
    movies_df = movies_df.drop(columns=['release_date', 'video_release_date', 'IMDb_URL'])

    ratings_df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = ratings_df.drop(columns=['timestamp'])

    return movies_df, ratings_df
