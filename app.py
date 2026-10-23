import streamlit as st
import pandas as pd
from model import load_data, build_user_item_matrix, train_matrix_factorization, get_recommendations

st.set_page_config(
    page_title="Intern Learning Path Recommendation",
    page_icon="🎓",
    layout="centered"
)

# Professional Aesthetic Styling (Clean Light Theme with Indigo Accents)
st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Inter', sans-serif;
    }
    .stSidebar {
        background-color: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #0f172a !important;
    }
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        padding: 30px;
        border-radius: 12px;
        color: white !important;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.1);
    }
    .main-header h1, .main-header p {
        color: white !important;
    }
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        text-align: center;
    }
    .recommendation-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.2s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def get_cached_data():
    return load_data()

interactions_df, courses_df = get_cached_data()
pivot_df = build_user_item_matrix(interactions_df)

@st.cache_resource
def get_cached_model(pivot):
    return train_matrix_factorization(pivot)

predicted_ratings_df = get_cached_model(pivot_df)

# Aesthetic Header Banner
st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size: 26px;">🎓 Intern Learning Path Recommender</h1>
        <p style="margin:5px 0 0 0; opacity: 0.9; font-size: 14px;">AI-Driven Personalized Curriculum Generation via Collaborative Filtering</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
selected_intern = st.sidebar.selectbox("Select Intern ID", options=sorted(interactions_df['intern_id'].unique()))
n_recommendations = st.sidebar.slider("Modules to Suggest", min_value=3, max_value=8, value=5)
selected_category = st.sidebar.selectbox("Filter Category", options=["All"] + list(courses_df['category'].unique()))

# Quick Summary Metrics
completed_courses = interactions_df[interactions_df['intern_id'] == selected_intern]
avg_score = completed_courses['completion_score'].mean() if len(completed_courses) > 0 else 0

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
        <div class="metric-container">
            <span style="font-size: 13px; color: #64748b !important; font-weight: 600;">COMPLETED MODULES</span>
            <h2 style="margin: 5px 0 0 0; color: #4f46e5 !important;">{len(completed_courses)}</h2>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="metric-container">
            <span style="font-size: 13px; color: #64748b !important; font-weight: 600;">AVG PERFORMANCE SCORE</span>
            <h2 style="margin: 5px 0 0 0; color: #10b981 !important;">{avg_score:.2f} / 5.0</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Recommendations Display
st.subheader(f"🎯 Recommended Curriculum for `{selected_intern}`")
st.markdown("")

top_recommendations = get_recommendations(
    selected_intern, n_recommendations, selected_category, 
    interactions_df, courses_df, predicted_ratings_df
)

if len(top_recommendations) > 0:
    for idx, row in top_recommendations.reset_index(drop=True).iterrows():
        st.markdown(f"""
            <div class="recommendation-card">
                <div style="font-weight: 600; font-size: 16px; color: #1e293b;">{idx + 1}. {row['course_title']}</div>
                <div style="margin-top: 6px; font-size: 13px; color: #64748b;">
                    🏷️ <b>Category:</b> {row['category']} &nbsp;|&nbsp; 
                    📊 <b>Level:</b> {row['difficulty']} &nbsp;|&nbsp; 
                    ⏱️ <b>Time:</b> {row['estimated_hours']} hrs &nbsp;|&nbsp; 
                    ⭐ <b>Match Affinity:</b> <span style="color: #4f46e5; font-weight: 600;">{row['predicted_affinity']:.2f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("No recommendations found matching your filters.")