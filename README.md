# Customer Churn Prediction

## Project Overview

This project aims to predict customer churn for **RetailGenius**, a fictional e-commerce platform. By leveraging AI and machine learning models, the goal is to predict which customers are at risk of churning and take preventive actions to improve customer retention. The project covers the entire workflow from data preparation to model deployment.

## Project Structure

The project follows a modular approach, with clear separation between different stages of the machine learning pipeline. The main components of the project are:


### Data

- **Raw Data**: The raw customer interaction, transaction, and product data.
- **Processed Data**: Cleaned and transformed data, ready for model training and analysis.

### Notebooks

The notebooks contain:
- **Exploratory Data Analysis (EDA)**: Data visualizations, distributions, correlations, etc.
- **Model Evaluation**: A deeper dive into how various models perform.

### Source Code

- **`data_preprocessing.py`**: Contains scripts for data cleaning, feature extraction, and transformation.
- **`model_training.py`**: Handles model training and evaluation. It includes model selection, hyperparameter tuning, and performance metrics.
- **`model_inference.py`**: Provides functions to make predictions using the trained model.

### Requirements

To install the dependencies, you can create a virtual environment and install the necessary packages:

```bash
pip install -r requirements.txt


