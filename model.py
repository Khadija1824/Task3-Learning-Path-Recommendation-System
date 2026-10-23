import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os

def load_data():
    """Loads intern learning patterns and course metadata CSV files."""
    if os.path.exists("intern_learning_patterns.csv") and os.path.exists("course_metadata.csv"):
        interactions = pd.read_csv("intern_learning_patterns.csv")
        courses = pd.read_csv("course_metadata.csv")
    else:
        # Fallback synthetic data if files are missing
        np.random.seed(42)
        intern_ids = [f"INT_{i:03d}" for i in range(1, 21)]
        course_ids = [f"CRS_{i:03d}" for i in range(1, 11)]
        courses = pd.DataFrame({
            "course_id": course_ids,
            "course_title": [f"Module {i}: Python & ML" for i in range(1, 11)],
            "category": ["Python & Programming", "Data Science & ML"] * 5,
            "difficulty": ["Beginner", "Intermediate", "Advanced"] * 3 + ["Beginner"],
            "estimated_hours": np.random.randint(10, 30, size=10)
        })
        interactions = pd.DataFrame({
            "intern_id": np.random.choice(intern_ids, 50),
            "course_id": np.random.choice(course_ids, 50),
            "completion_score": np.random.choice([4, 5], 50)
        })
    return interactions, courses

def build_user_item_matrix(interactions_df):
    """Builds the pivot table (User-Item Matrix) from interactions."""
    return interactions_df.pivot_table(index='intern_id', columns='course_id', values='completion_score', fill_value=0)

def train_matrix_factorization(pivot_df):
    """Trains TruncatedSVD Matrix Factorization model and returns predicted ratings dataframe."""
    n_components = min(3, pivot_df.shape[0]-1, pivot_df.shape[1]-1)
    if n_components < 1:
        n_components = 1
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    matrix_transformed = svd.fit_transform(pivot_df)
    predicted_matrix = np.dot(matrix_transformed, svd.components_)
    
    predicted_df = pd.DataFrame(predicted_matrix, index=pivot_df.index, columns=pivot_df.columns)
    return predicted_df

def get_recommendations(intern_id, n_recommendations, selected_category, interactions_df, courses_df, predicted_ratings_df):
    """Generates personalized course recommendations for a given intern."""
    if intern_id not in predicted_ratings_df.index:
        return pd.DataFrame()
    
    user_preds = predicted_ratings_df.loc[intern_id]
    taken_courses = interactions_df[interactions_df['intern_id'] == intern_id]['course_id'].tolist()
    
    unseen_preds = user_preds.drop(labels=taken_courses, errors='ignore')
    recommended_course_ids = unseen_preds.sort_values(ascending=False).index.tolist()
    
    rec_df = courses_df[courses_df['course_id'].isin(recommended_course_ids)].copy()
    rec_df['predicted_affinity'] = rec_df['course_id'].map(unseen_preds)
    rec_df = rec_df.sort_values(by='predicted_affinity', ascending=False)
    
    if selected_category != "All":
        rec_df = rec_df[rec_df['category'] == selected_category]
        
    return rec_df.head(n_recommendations)