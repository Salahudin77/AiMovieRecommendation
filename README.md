A Movie Recommendation System developed as part of the ID1214 - AI Course at KTH.
The system provides personalized movie recommendations using Collaborative Filtering, Genre-Based Filtering, and Deep Learning (Neural Collaborative Filtering - NCF).

🚀 Features
🎞 Hybrid Recommendation System (Genre-Based, Collaborative Filtering, and Deep Learning)
🧠 Neural Collaborative Filtering (NCF) with TensorFlow
🎭 Cold-Start Handling with Genre-Based Filtering
📊 Scalable and Efficient Model Training
💻 Interactive Streamlit UI for Recommendations

📂 Project Structure
📦 movie-recommendation-system
 ┣ 📂 data/                  # Dataset folder (u.data, u.item)
 ┣ 📂 models/                # Saved ML models (NMF, SVD, NCF)
 ┣ 📂 scripts/               # Core Python scripts
 ┃ ┣ 📜 utils.py             # Data preprocessing
 ┃ ┣ 📜 collaborative_filtering.py  # NMF & SVD
 ┃ ┣ 📜 neural_cf.py         # Neural Collaborative Filtering (Deep Learning)
 ┃ ┣ 📜 movieRecommendation.py  # Genre-Based Filtering
 ┃ ┣ 📜 app.py               # Streamlit UI for recommendations
 ┣ 📜 requirements.txt        # Required dependencies
 ┣ 📜 README.md               # Project Documentation

 🛠️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit Application
streamlit run app.py

🎯 Usage
Open the Streamlit UI.
Select your favorite genres for recommendations OR Enter User ID for Collaborative Filtering.
Click "Get Recommendations" to view movie suggestions.

🧠 AI Techniques Used
✅ Non-Negative Matrix Factorization (NMF) – Collaborative Filtering
✅ Singular Value Decomposition (SVD) – Collaborative Filtering
✅ Neural Collaborative Filtering (NCF) – Deep Learning for recommendations
✅ Genre-Based Filtering – Content-based approach
✅ Hybrid System Integration – Combining all methods
