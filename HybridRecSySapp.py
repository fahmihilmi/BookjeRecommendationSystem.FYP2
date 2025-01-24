import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import random

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
    
    # Check for image_url column
    if 'image_url' not in df.columns:
        df['image_url'] = ''  # Fallback if no image_url column

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
def recommend_random_good_items(df, top_n=5):
    # Filter "good" listings based on review rate number (e.g., reviews >= 4)
    good_items = df[df['review rate number'].astype(float) >= 4]
    
    # Randomly pick 'top_n' listings from the good items
    recommended = good_items.sample(n=top_n, random_state=42)
    return recommended[['id', 'NAME', 'room type', 'neighbourhood group', 'review rate number']]

# Sidebar Navigation
page = sidebar_navigation()

# Handle Navigation Pages
if page == "Home":
    st.title("Airbnb Hybrid Recommendation System")
    
    # Add a dummy picture
    st.image("https://via.placeholder.com/600x300", caption="Welcome to the Airbnb Recommendation System", use_container_width=True)
    
    # Recommend random "good" listings
    st.write("Here are some top recommendations for you:")
    try:
        recommendations = recommend_random_good_items(df, top_n=5)

        # Display random "good" recommendations
        for _, row in recommendations.iterrows():
            st.markdown(f"### {row['NAME']}")
            st.markdown(f"**Room Type:** {row['room type']} | **Neighborhood:** {row['neighbourhood group']}")
            st.markdown(f"**Review Rate Number:** {row['review rate number']}")
            st.markdown("---")

    except Exception as e:
        st.error(f"Error generating recommendations: {e}")

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
