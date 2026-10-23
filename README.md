Learning Path Recommendation System

An AI-driven, modular machine learning pipeline and interactive web application designed to recommend personalized learning modules and courses for interns using Collaborative Filtering (Matrix Factorization).

Project Overview

Managing professional development and curating targeted upskilling paths for interns at scale can be challenging. This project builds a collaborative filtering recommendation engine that analyzes past intern learning patterns, course completion scores, and metadata to predict and suggest optimal custom learning paths.

Key Features

Matrix Factorization Engine: Utilizes Singular Value Decomposition (TruncatedSVD) to uncover latent relationships between interns and courses, accurately predicting unrated or uncompleted course affinities.

Modular Architecture: Clean separation of concerns between machine learning logic (model.py) and the user interface (app.py).

Interactive Streamlit UI: A professional, enterprise-grade dashboard featuring customizable filters, real-time recommendation scoring, and performance summary metrics.

Dataset Description

1. intern_learning_patterns.csv

Contains historical interaction data tracking which courses interns have taken and their respective performance/completion ratings.

intern_id: Unique identifier for each intern (e.g., INT_001)

course_id: Unique identifier for each course (e.g., CRS_001)

completion_score: Qualitative score or rating awarded upon completion (Scale: 1 to 5)

2. course_metadata.csv

Contains descriptive metadata for the entire course library.

course_id: Unique course identifier

course_title: Title and description of the learning module

category: Domain area (e.g., Python & Programming, Data Science & ML)

difficulty: Skill level (Beginner, Intermediate, Advanced)

estimated_hours: Expected duration required to complete the module

Machine Learning Pipeline

User-Item Interaction Matrix: The pipeline pivots the interaction data to construct a sparse User-Item Matrix where rows represent interns (intern_id), columns represent courses (course_id), and values represent completion scores.

Matrix Factorization (TruncatedSVD): Decomposes the high-dimensional sparse matrix into lower-dimensional latent feature spaces, learning hidden patterns in intern learning preferences.

Affinity Prediction & Filtering: Reconstructs the rating matrix to predict affinity scores for courses the intern has not yet taken, ranking them in descending order of relevance.

Setup & Installation

To run this project locally, ensure you have Python installed along with the required libraries.

1. Install Dependencies

pip install streamlit pandas numpy scikit-learn


2. Verify Files

Ensure model.py, app.py, intern_learning_patterns.csv, and course_metadata.csv are placed in the same working directory.

3. Run the Streamlit Application

Execute the following command in your terminal:

streamlit run app.py
