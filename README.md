**End-to-End Movie Revenue Predictor**
An interactive machine learning application that predicts the worldwide box office revenue of a movie based on its budget, genre, cast, and more. 
This project covers the entire data science workflow from data cleaning and feature engineering to model training and deployment.

**Project Overview**
This project aims to build a robust regression model to forecast movie success. The process involved several key stages:

Data Collection: Used the "TMDB 5000 Movie Dataset" from Kaggle, which contains information on over 5,000 movies.

Data Cleaning & EDA: Processed and cleaned the data using Pandas. Handled missing values, removed irrelevant entries (e.g., movies with a $0 budget), and performed exploratory data analysis (EDA) with Seaborn to uncover initial trends.

Feature Engineering: Extracted and transformed key features for the model. This included:

Parsing complex JSON columns (cast, crew, genres).

Extracting key information like the lead actor and director.

One-hot encoding categorical features to convert them into a numerical format.

Modeling: Trained and evaluated several regression models using Scikit-learn. The final model is a fine-tuned RandomForestRegressor which was selected for its performance and interpretability.

Deployment: Built an interactive web application using Gradio and deployed it on Hugging Face Spaces for public access.

**Key Findings & Model Performance**
The final model predicts movie revenue with a high degree of accuracy, achieving an R-squared score of approximately 0.78 on unseen test data.

Budget and Popularity were identified as the two most significant predictors of a movie's box office revenue.

Feature engineering was critical: including the director and lead actor as features noticeably improved the model's predictive power.

Technical Stack
Programming Language: Python 3

Libraries:

Data Manipulation: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost

Web App & Deployment: Gradio, Hugging Face Spaces

Development Environment: Google Colab

**How to Run This Project Locally**
To run the interactive Gradio app on your own machine, follow these steps:

Clone the repository:

git clone [https://github.com/YourUsername/YourRepository.git](https://github.com/karanrp813/Movie-Revenue-Predictor.git)

cd YourRepository

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

Run the application:

python app.py

A local URL will be generated, which you can open in your browser to use the app.

Repository File Structure
├── app.py                      # The Python script for the Gradio web application
├── movie_revenue_predictor.pkl   # The final, saved Scikit-learn model
├── MovieSuccessPredictor.ipynb   # The main Jupyter Notebook with all analysis and model training
├── requirements.txt              # A list of all Python libraries required to run the project
└── README.md                     # This file!
