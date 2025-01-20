import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data from utils
from utils import load_data
movies_df, ratings_df = load_data()

# Prepare data for NCF
def prepare_ncf_data(ratings_df):
    user_ids = ratings_df['user_id'].unique()
    movie_ids = ratings_df['movie_id'].unique()

    user_id_map = {id_: i for i, id_ in enumerate(user_ids)}
    movie_id_map = {id_: i for i, id_ in enumerate(movie_ids)}

    ratings_df['user_idx'] = ratings_df['user_id'].map(user_id_map)
    ratings_df['movie_idx'] = ratings_df['movie_id'].map(movie_id_map)

    return ratings_df, len(user_ids), len(movie_ids)

ratings_df, num_users, num_movies = prepare_ncf_data(ratings_df)

# Split data for training and testing
train, test = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Define the NCF model
def build_ncf_model(num_users, num_movies, embedding_dim=50):
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    # Embedding layers
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
    movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_dim, name='movie_embedding')(movie_input)

    # Flatten embeddings
    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)

    # Concatenate user and movie vectors
    concat = Concatenate()([user_vec, movie_vec])

    # Fully connected layers
    dense_1 = Dense(128, activation='relu')(concat)
    dense_2 = Dense(64, activation='relu')(dense_1)
    output = Dense(1, activation='linear')(dense_2)  # Predicting ratings

    # Build model
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

# Build the model
ncf_model = build_ncf_model(num_users, num_movies)

# Prepare training data
X_train = [train['user_idx'].values, train['movie_idx'].values]
y_train = train['rating'].values

X_test = [test['user_idx'].values, test['movie_idx'].values]
y_test = test['rating'].values

# Train the model
ncf_model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
ncf_model.save("ncf_model.h5")

# Generate recommendations
def recommend_ncf(user_id, top_n=5):
    # Check if user exists
    user_map = {v: k for k, v in enumerate(ratings_df['user_id'].unique())}
    if user_id not in user_map:
        raise ValueError("User not found in the dataset")

    user_idx = user_map[user_id]
    movie_indices = np.arange(num_movies)

    # Predict ratings for all movies for the given user
    user_input = np.array([user_idx] * num_movies)
    predictions = ncf_model.predict([user_input, movie_indices]).flatten()

    # Get top N recommendations
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    recommendations = movies_df[movies_df['movie_id'].isin(top_indices)]

    return recommendations[['title', 'movie_id']].to_dict('records')

# Feedback loop to update the NCF model
def update_ncf_model(new_ratings, model_path="ncf_model.h5"):
    """
    Update the NCF model with new user ratings.
    
    Parameters:
        new_ratings (pd.DataFrame): A DataFrame with columns ['user_id', 'movie_id', 'rating'].
        model_path (str): Path to the saved NCF model.

    Returns:
        Updated NCF model.
    """
    global ratings_df, ncf_model

    # Map user and movie IDs to indices
    user_ids = ratings_df['user_id'].unique()
    movie_ids = ratings_df['movie_id'].unique()
    user_id_map = {id_: i for i, id_ in enumerate(user_ids)}
    movie_id_map = {id_: i for i, id_ in enumerate(movie_ids)}

    # Add new ratings
    new_ratings['user_idx'] = new_ratings['user_id'].map(user_id_map)
    new_ratings['movie_idx'] = new_ratings['movie_id'].map(movie_id_map)
    ratings_df = pd.concat([ratings_df, new_ratings], ignore_index=True)

    # Prepare training data
    X_train = [ratings_df['user_idx'].values, ratings_df['movie_idx'].values]
    y_train = ratings_df['rating'].values

    # Load the existing model
    ncf_model = tf.keras.models.load_model(model_path)

    # Retrain the model with new data
    ncf_model.fit(x=X_train, y=y_train, batch_size=32, epochs=3)  # Fewer epochs for quick updates

    # Save the updated model
    ncf_model.save(model_path)
    print("NCF model updated successfully.")

    return ncf_model

# Calculate RMSE
def calculate_rmse(y_true, y_pred):
    """
    Calculate RMSE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Evaluate the NCF model
y_pred = ncf_model.predict(X_test).flatten()
rmse = calculate_rmse(y_test, y_pred)
print(f"NCF RMSE: {rmse}")
