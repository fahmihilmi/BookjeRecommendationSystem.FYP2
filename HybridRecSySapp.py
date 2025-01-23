import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load Dataset from GitHub
@st.cache_data
def load_data_from_github(file_url):
    try:
        # Read the CSV file directly from the GitHub raw URL
        return pd.read_csv(file_url)
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        return None

# Correct GitHub raw URL for Minimized_Airbnb_Data.csv
GITHUB_RAW_URL = 'https://raw.githubusercontent.com/fahmihilmi/BookjeRecommendationSystem.FYP2/main/Minimized_Airbnb_Data.csv'

df = load_data_from_github(GITHUB_RAW_URL)
if df is None:
    st.stop()  # Stop the app if the data cannot be loaded

# Preprocess Data
def preprocess_data(df):
    # Fill missing values
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

# Correct Spelling
def correct_spelling(value, correct_values):
    if isinstance(value, str) and value.strip():
        best_match = process.extractOne(value, correct_values)
        return best_match[0] if best_match[1] > 80 else value
    return None

correct_values = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']
df['neighbourhood group'] = df['neighbourhood group'].apply(lambda x: correct_spelling(x, correct_values))

# Train Model
@st.cache_data
def train_models(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    svd = TruncatedSVD(n_components=100, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    return tfidf_matrix, svd, svd_matrix

tfidf_matrix, svd, svd_matrix = train_models(df)

def recommend_hybrid(listing_id, tfidf_matrix, svd_model, df, svd_matrix, alpha=0.5, top_n=5):
    # Calculate content-based similarity (TF-IDF)
    content_sim = cosine_similarity(tfidf_matrix[listing_id], tfidf_matrix).flatten()

    # Calculate collaborative similarity (SVD matrix)
    collaborative_sim = cosine_similarity(svd_matrix[listing_id].reshape(1, -1), svd_matrix).flatten()

    # Compute hybrid scores
    hybrid_scores = alpha * content_sim + (1 - alpha) * collaborative_sim

    # Sort and get top N recommendations
    sorted_indices = hybrid_scores.argsort()[::-1]
    recommended = df.iloc[sorted_indices[1:top_n + 1]]  # Skip the first one, as it's the same listing
    return recommended[['id', 'NAME', 'room type', 'neighbourhood group', 'review rate number']]

# Streamlit App Layout
st.title("Airbnb Hybrid Recommendation System")

# User Input
user_input = st.text_input("Enter the name of a listing or keyword:")

if user_input:
    # Fuzzy matching to find the closest match
    def find_closest_match(user_input, df, column="NAME"):
        best_match = process.extractOne(user_input, df[column].dropna().tolist())
        if best_match and best_match[1] > 80:  # Confidence threshold
            return best_match[0]
        return None

    matched_name = find_closest_match(user_input, df, column="NAME")

    if matched_name:
        st.write(f"Best match found: {matched_name}")

        # Get the index of the matched listing
        matched_index = df[df["NAME"] == matched_name].index[0]

        # Generate recommendations
        recommended = recommend_hybrid(
            listing_id=matched_index,
            tfidf_matrix=tfidf_matrix,
            svd_model=svd,
            df=df,
            svd_matrix=svd_matrix,
            alpha=0.5,
            top_n=5
        )
        st.write("Recommended Listings:")
        st.table(recommended)
    else:
        st.write("No matching listing found. Please try again.")
