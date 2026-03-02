# E-commerce-Customer-Supprt
Predicting Customer Satisfaction using Advanced Machine Learning &amp; Bayesian Optimization

E-Commerce CSAT Intelligence System

Predictive Customer Satisfaction Modeling with XGBoost & Bayesian Optimization

Project Overview

In the fast-paced world of E-commerce, waiting for a customer to fill out a survey is a reactive strategy. This project introduces a proactive approach by predicting Customer Satisfaction (CSAT) scores (1–5) based on 85,000+ support interactions. By analyzing the "Digital Body Language" of a customer—their remarks, order categories, and interaction metadata—this system identifies frustrated customers before they even submit a review.

Key Features
Large-Scale NLP: Processed over 85,000 rows of unstructured text using TF-IDF vectorization.

Advanced Optimization: Utilized Bayesian Search (Optuna) to find the optimal hyperparameters for XGBoost, outperforming standard Grid Search.

Handling Imbalance: Implemented SMOTE to ensure the model accurately detects high-risk 1-star and 2-star ratings.

Explainable AI (XAI): Integrated SHAP values to decode why the model predicts a certain score, identifying key business friction points.

Tech Stack
Language: Python 3.10+

Core ML: XGBoost, Scikit-Learn

NLP: NLTK, TF-IDF

Optimization: Optuna (Bayesian Optimization)

Interpretability: SHAP

Deployment Ready: Joblib for model persistence

Model Performance
The final Optimized XGBoost model achieved a significant performance lift during training:

Baseline Accuracy: 32.4%

Optimized Accuracy: 37.4% (~5% absolute improvement)

Business Impact: Correctly categorized thousands of additional "At-Risk" customers compared to baseline models.

