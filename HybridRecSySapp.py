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
GITHUB_RAW_URL = 'https://raw.githubusercontent.com/fahmihilmi/BookjeRecommendationSystem.FYP2/main/Minimized_Airbnb_Data.csv'  # Corrected URL

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
    return tfidf_matrix, svd

tfidf_matrix, svd = train_models(df)

# Hybrid Recommendation Function
def recommend_hybrid(listing_id, tfidf_matrix, svd_model, df, alpha=0.5, top_n=5):
    # Calculate content-based similarity (TF-IDF)
    content_sim = cosine_similarity(tfidf_matrix[listing_id], tfidf_matrix).flatten()

    # Ensure collaborative similarity has the same shape as content_sim
    latent_features = svd_model.transform(tfidf_matrix[listing_id])
    collaborative_sim = svd_model.inverse_transform(latent_features).flatten()

    # Ensure both are 1-dimensional arrays and have the same length
    if content_sim.ndim > 1:
        content_sim = content_sim.flatten()
    if collaborative_sim.ndim > 1:
        collaborative_sim = collaborative_sim.flatten()

    # Check if lengths match, truncate if necessary
    if len(content_sim) != len(collaborative_sim):
        # Optionally truncate or align them in a way that makes sense for your dataset
        collaborative_sim = collaborative_sim[:len(content_sim)]

    # Compute hybrid scores
    hybrid_scores = alpha * content_sim + (1 - alpha) * collaborative_sim

    # Sort and get top N recommendations
    sorted_indices = hybrid_scores.argsort()[::-1]
    recommended = df.iloc[sorted_indices[1:top_n + 1]]  # Skip the first one, as it's the same listing
    return recommended[['id', 'NAME', 'room type', 'neighbourhood group', 'review rate number']]

# Streamlit App Layout
st.title("Airbnb Hybrid Recommendation System")

# User Input
selected_neighbourhood = st.selectbox("Select a Neighbourhood Group", df['neighbourhood group'].unique())
selected_room_type = st.selectbox("Select a Room Type", df['room type'].unique())

# Filter and Recommend
filtered_df = df[(df['neighbourhood group'] == selected_neighbourhood) & (df['room type'] == selected_room_type)]

# Optionally, you can load recommendations by default when the page loads
if filtered_df.empty:
    st.write("No listings found with your initial selection.")
else:
    # Load default recommendations (first row of the filtered list)
    listing_idx = filtered_df.index[0]
    recommended = recommend_hybrid(
        listing_id=listing_idx,
        tfidf_matrix=tfidf_matrix,
        svd_model=svd,
        df=df,
        alpha=0.5,
        top_n=5
    )
    st.write("Recommended Listings on Page Load:")
    st.table(recommended)

# User can click to get new recommendations
if st.button("Recommend Listings"):
    if filtered_df.empty:
        st.write("No listings found with your selections.")
    else:
        listing_idx = filtered_df.index[0]
        recommended = recommend_hybrid(
            listing_id=listing_idx,
            tfidf_matrix=tfidf_matrix,
            svd_model=svd,
            df=df,
            alpha=0.5,
            top_n=5
        )
        st.write("Recommended Listings:")
        st.table(recommended)
