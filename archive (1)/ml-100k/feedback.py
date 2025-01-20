import streamlit as st
import pandas as pd

# Example recommendations
recommendations = [
    {"movie_id": 1, "title": "Movie A"},
    {"movie_id": 2, "title": "Movie B"},
    {"movie_id": 3, "title": "Movie C"}
]

# Display recommendations
st.title("Movie Recommendation System")
st.write("### Your Recommendations")
feedback_data = []

for movie in recommendations:
    st.write(f"**{movie['title']}** (ID: {movie['movie_id']})")
    feedback = st.radio(
        f"Did you like {movie['title']}?",
        options=["No feedback", "Like", "Dislike"],
        key=f"feedback_{movie['movie_id']}"
    )
    feedback_data.append({"movie_id": movie["movie_id"], "title": movie["title"], "feedback": feedback})

# Save feedback
if st.button("Submit Feedback"):
    feedback_df = pd.DataFrame(feedback_data)
    feedback_df.to_csv("user_feedback.csv", index=False)
    st.success("Feedback submitted! Thank you.")

# Optional: Display feedback
st.write("### Submitted Feedback")
st.write(feedback_data)
