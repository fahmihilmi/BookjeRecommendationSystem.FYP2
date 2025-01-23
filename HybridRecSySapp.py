import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Add custom page config
st.set_page_config(
    page_title="Airbnb Recommendation System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Header
st.sidebar.header("📌 Navigation")
st.sidebar.markdown("""
- **Home**: Learn about the recommendation system.
- **Search**: Enter a keyword or listing name.
- **Recommendations**: Get tailored results.
""")

# Load Dataset from GitHub
@st.cache_data
def load_data_from_github(file_url):
    try:
        return pd.read_csv(file_url)
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        return None

# GitHub raw dataset URL
GITHUB_RAW_URL = "https://raw.githubusercontent.com/fahmihilmi/BookjeRecommendationSystem.FYP2/main/Minimized_Airbnb_Data.csv"

# Load the dataset
df = load_data_from_github(GITHUB_RAW_URL)
if df is None:
    st.stop()

# Preprocess the data
def preprocess_data(df):
    df['NAME'] = df['NAME'].fillna('')
    df['host_identity_verified'] = df['host_identity_verified'].fillna('')
    df['neighbourhood group'] = df['neighbourhood group'].fillna('')
    df['review rate number'] = df['review rate number'].fillna('0').astype(str)

    # Combine features for TF-IDF
    df['combined_features'] = (
        df['NAME'] + ' ' +
        df['host_identity_verified'] + ' ' +
        df['neighbourhood group'] + ' ' +
        df['review rate number']
    )
    return df

df = preprocess_data(df)

# Correct Spelling Function
def correct_spelling(value, correct_values):
    if isinstance(value, str) and value.strip():
        best_match = process.extractOne(value, correct_values)
        return best_match[0] if best_match[1] > 80 else value
    return None

# Correct spelling for 'neighbourhood group'
correct_values = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']
df['neighbourhood group'] = df['neighbourhood group'].apply(lambda x: correct_spelling(x, correct_values))

# Train Models
@st.cache_data
def train_models(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    svd = TruncatedSVD(n_components=100, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    return tfidf_matrix, svd, svd_matrix

tfidf_matrix, svd, svd_matrix = train_models(df)

# Recommendation Function
def recommend_hybrid(name_input, tfidf_matrix, svd_matrix, df, alpha=0.5, top_n=5):
    # Fuzzy matching to find the closest match in the dataset
    matched_name = process.extractOne(name_input, df['NAME'].dropna().tolist())
    if not matched_name or matched_name[1] < 80:
        return None, "No close match found for your input."

    # Get the index of the matched name
    matched_index = df[df['NAME'] == matched_name[0]].index[0]

    # Content-based similarity
    content_sim = cosine_similarity(tfidf_matrix[matched_index], tfidf_matrix).flatten()

    # Collaborative similarity
    collaborative_sim = cosine_similarity(svd_matrix[matched_index].reshape(1, -1), svd_matrix).flatten()

    # Compute hybrid scores
    hybrid_scores = alpha * content_sim + (1 - alpha) * collaborative_sim

    # Sort and get top N recommendations
    sorted_indices = hybrid_scores.argsort()[::-1]
    recommended = df.iloc[sorted_indices[1:top_n + 1]]  # Skip the first one (self)
    return recommended, matched_name[0]

# Streamlit Home Page Layout
st.title("🏠 Airbnb Hybrid Recommendation System")
st.markdown("""
Welcome to the **Airbnb Hybrid Recommendation System**! 🎉  
Discover personalized Airbnb listings based on your preferences.

### How It Works
- **Input**: Enter a listing name or keyword below.
- **Recommendations**: See personalized results based on a hybrid of content-based and collaborative filtering.

---

### Try It Out
👉 Type a listing name or keyword below to get started.
""")

# Input box for user search
user_input = st.text_input("Enter a listing name or keyword:")

# Process user input and show recommendations
if user_input:
    recommendations, matched_name = recommend_hybrid(
        name_input=user_input,
        tfidf_matrix=tfidf_matrix,
        svd_matrix=svd_matrix,
        df=df,
        alpha=0.5,
        top_n=5
    )

    if recommendations is None:
        st.warning(f"⚠️ {matched_name}")
    else:
        st.success(f"✅ Showing recommendations for: {matched_name}")
        st.table(recommendations[['id', 'NAME', 'neighbourhood group', 'room type', 'review rate number']])

# Footer
st.markdown("---
