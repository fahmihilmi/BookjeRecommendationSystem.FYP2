import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load your dataset
@st.cache
def load_data():
    return pd.read_csv('Airbnb_Open_Data.csv')

df = load_data()

# Preprocess your data
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

# Correct spelling using fuzzy matching
def correct_spelling(value, correct_values):
    if isinstance(value, str) and value.strip():
        best_match = process.extractOne(value, correct_values)
        return best_match[0] if best_match[1] > 80 else value
    return None

correct_values = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']
df['neighbourhood group'] = df['neighbourhood group'].apply(lambda x: correct_spelling(x, correct_values))

# Train TF-IDF and SVD models
@st.cache
def train_models(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    svd = TruncatedSVD(n_components=100, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    return tfidf_matrix, svd

tfidf_matrix, svd = train_models(df)

# Recommendation function
def recommend_hybrid(listing_id, tfidf_matrix, svd_model, df, alpha=0.5, top_n=5):
    content_sim = cosine_similarity(tfidf_matrix[listing_id], tfidf_matrix).flatten()
    latent_features = svd_model.transform(tfidf_matrix[listing_id])
    collaborative_sim = svd_model.inverse_transform(latent_features).flatten()
    hybrid_scores = alpha * content_sim + (1 - alpha) * collaborative_sim

    sorted_indices = hybrid_scores.argsort()[::-1]
    recommended = df.iloc[sorted_indices[1:top_n + 1]]
    return recommended[['id', 'NAME', 'room type', 'neighbourhood group', 'review rate number']]

# Streamlit app layout
st.title("Airbnb Hybrid Recommendation System")

# User input
selected_neighbourhood = st.selectbox("Select a Neighbourhood Group", df['neighbourhood group'].unique())
selected_room_type = st.selectbox("Select a Room Type", df['room type'].unique())

# Filter and recommend
filtered_df = df[(df['neighbourhood group'] == selected_neighbourhood) & (df['room type'] == selected_room_type)]

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
