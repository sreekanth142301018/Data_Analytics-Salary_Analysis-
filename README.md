# Salary Analysis and Prediction System

## Overview
This project is a web-based data analytics application built using Flask that analyzes salary datasets to extract insights, visualize trends, predict salaries using machine learning models, and generate association rules between skills. It integrates data preprocessing, exploratory data analysis, predictive modeling, and rule mining into a single interactive system.

## Features
- **Data Preprocessing**
  - Handles missing values (mean for numerical, mode for categorical)
  - Removes high-cardinality columns (>300 unique values)
  - Automatically detects numerical, categorical, and skill-based features

- **Exploratory Data Analysis (EDA)**
  - Interactive visualizations using Plotly:
    - Bar charts
    - Pie charts
    - Box plots
    - Histograms
    - Scatter plots
  - Analysis across departments, skills, and salary distributions

- **Salary Prediction (Machine Learning)**
  - Dynamic target and feature selection
  - Categorical feature encoding using Label Encoding
  - Train-test split (80/20)
  - Model comparison using:
    - Linear Regression
    - Ridge Regression
    - Support Vector Regression (SVR)
    - Random Forest Regressor
  - Performance evaluation using:
    - Mean Squared Error (MSE)
    - R² Score
  - Actual vs Predicted salary visualization

- **Association Rule Mining**
  - Uses Apriori algorithm to discover frequent skill combinations
  - Generates association rules based on lift metric
  - Predicts potential salary improvement by learning new skill combinations
  - Recommends top skill boosts based on expected salary increase

## Technologies Used
- **Backend:** Python, Flask
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Plotly
- **Association Rules:** mlxtend (Apriori, association_rules)

## Project Structure
