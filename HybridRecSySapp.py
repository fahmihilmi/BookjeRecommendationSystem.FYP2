import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Sidebar Navigation
def sidebar_navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Profile", "Settings", "Messages"]
    )
    return page

# Load Dataset from GitHub
@st.cache_data
def load_data_from_github(file_url):
    try:
        return pd.read_csv(file_url)
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        return None

# Correct GitHub raw URL for Minimized_Airbnb_Data.csv
GITHUB_RAW_URL = 'https://raw.githubusercontent.com/fahmihilmi/BookjeRecommendationSystem.FYP2/main/Minimized_Airbnb_Data.csv'

df = load_data_from_github(GITHUB_RAW_URL)
if df is None:
    st.stop()

# Preprocess Data
def preprocess_data(df):
    df['NAME'] = df['NAME'].fillna('')
    df['host_identity_verified'] = df['host_identity_verified'].fillna('')
    df['neighbourhood group'] = df['neighbourhood group'].fillna('')
    df['review rate number'] = df['review rate number'].fillna('0').astype(str)
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

# Recommendation Function
def recommend_hybrid(listing_id, tfidf_matrix, svd_model, df, svd_matrix, alpha=0.5, top_n=5):
    content_sim = cosine_similarity(tfidf_matrix[listing_id], tfidf_matrix).flatten()
    collaborative_sim = cosine_similarity(svd_matrix[listing_id].reshape(1, -1), svd_matrix).flatten()
    hybrid_scores = alpha * content_sim + (1 - alpha) * collaborative_sim
    sorted_indices = hybrid_scores.argsort()[::-1]
    recommended = df.iloc[sorted_indices[1:top_n + 1]]
    return recommended[['id', 'NAME', 'room type', 'neighbourhood group', 'review rate number']]

# Handle Navigation
page = sidebar_navigation()

if page == "Home":
    st.title("Airbnb Hybrid Recommendation System")
    selected_neighbourhood = st.selectbox("Select a Neighbourhood Group", df['neighbourhood group'].unique())
    selected_room_type = st.selectbox("Select a Room Type", df['room type'].unique())

    filtered_df = df[(df['neighbourhood group'] == selected_neighbourhood) & (df['room type'] == selected_room_type)]
    if filtered_df.empty:
        st.write("No listings found with your initial selection.")
    else:
        listing_idx = filtered_df.index[0]
        recommended = recommend_hybrid(
            listing_id=listing_idx,
            tfidf_matrix=tfidf_matrix,
            svd_model=svd,
            df=df,
            svd_matrix=svd_matrix,
            alpha=0.5,
            top_n=5
        )
        st.write("Recommended Listings:")
        st.table(recommended)

elif page == "Profile":
    st.title("Profile Page")
    st.write("This is your profile page. You can display user information here.")

elif page == "Settings":
    st.title("Settings Page")
    st.write("This is the settings page where users can customize their preferences.")

elif page == "Messages":
    st.title("Messages Page")
    st.write("This is the messages page where users can see their messages or notifications.")

# Footer
st.markdown("---")
st.markdown("""
💡 **Tip**: Use specific keywords like `Luxury Loft` or `Manhattan`.  
For feedback, email us at **support@airbnb-recsys.com**.
""")
