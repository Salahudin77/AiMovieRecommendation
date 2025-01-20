# movieRecomendation.py
from utils import load_data  # Use the centralized load_data function

# Load data
movies_df, ratings_df = load_data()

# Precompute average ratings for efficiency
avg_ratings = ratings_df.groupby('movie_id')['rating'].mean()
rating_counts = ratings_df.groupby('movie_id')['rating'].count()
avg_ratings = avg_ratings[rating_counts >= 5]  # Movies with at least 5 ratings

# Genre mapping
# Map user-selected genres to corresponding columns in movies_df
genre_map = {
    "Action": 0,
    "Adventure": 1,
    "Animation": 2,
    "Children's": 3,
    "Comedy": 4,
    "Crime": 5,
    "Documentary": 6,
    "Drama": 7,
    "Fantasy": 8,
    "Film-Noir": 9,
    "Horror": 10,
    "Musical": 11,
    "Mystery": 12,
    "Romance": 13,
    "Sci-Fi": 14,
    "Thriller": 15,
    "War": 16,
    "Western": 17,
}

# Map selected genres to actual column names
def recommend_by_genre(favorite_genres, num_recommendations=5):
    if not favorite_genres:
        return []  # No genres selected, return empty recommendations.

    genre_columns = [f"genre_{genre_map[genre]}" for genre in favorite_genres if genre in genre_map]
    if not genre_columns:
        return []

    genre_filtered_movies = movies_df[movies_df[genre_columns].any(axis=1)]
    if genre_filtered_movies.empty:
        return []

    avg_ratings = ratings_df.groupby('movie_id')['rating'].mean()
    genre_filtered_movies = genre_filtered_movies.merge(avg_ratings, on='movie_id', how='inner')
    genre_filtered_movies = genre_filtered_movies.sort_values(by='rating', ascending=False)
    recommendations = genre_filtered_movies.head(num_recommendations)[['movie_id', 'title']].to_dict('records')
    return recommendations


def recommend_new_movies(movies_df, ratings_df, favorite_genres, num_recommendations=5):
    genre_columns = [f'genre_{genre_map[genre]}' for genre in favorite_genres if genre in genre_map]
    genre_filtered_movies = movies_df[movies_df[genre_columns].any(axis=1)]
    rated_movies = ratings_df['movie_id'].unique()
    new_movies = genre_filtered_movies[~genre_filtered_movies['movie_id'].isin(rated_movies)]
    if new_movies.empty:
        return recommend_by_genre(movies_df, ratings_df, favorite_genres, num_recommendations)
    recommendations = new_movies.head(num_recommendations)['title'].tolist()
    return recommendations
