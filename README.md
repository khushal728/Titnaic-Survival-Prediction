# Titnaic-Survival-Prediction

This repository contains a machine learning project to predict whether a passenger survived the Titanic disaster based on features like age, gender, class, and more. The project involves data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Table of Contents

1. Project Overview
2. Dataset
3. Requirements
4. Steps to Run the Code
5. Model Training and Evaluation
6. Project Structure
7. Key Results
8. Future Improvements
9. Acknowledgments


## Project Overview
The goal of this project is to:

Analyze the Titanic dataset.
Build a Logistic Regression model to predict survival outcomes.
Evaluate the model’s performance using various metrics like accuracy, classification report, and ROC-AUC.

## Dataset
The dataset (train_and_test2.csv) contains passenger information, including features such as age, gender, fare, class, and more. The target variable is 2urvived:

1: Passenger survived.

0: Passenger did not survive.

## Key Features:
Age: Age of the passenger.

Sex: Gender of the passenger.

Fare: Ticket fare.

Embarked: Port of embarkation.

Pclass: Passenger class (1st, 2nd, 3rd).

## Requirements
To run this project, you need the following Python libraries:

pandas

numpy

seaborn

matplotlib

scikit-learn

imblearn (for handling class imbalance)

joblib (for saving the model)


Install the required libraries using:

```pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib```

## Steps to Run the Code

1. Download the Dataset:

Save the train_and_test2.csv file in the project directory.

2. Run the Python Code:

Open the provided Python script in an IDE or Jupyter Notebook.
Ensure the dataset path is correctly specified in the script:

```file_path = "train_and_test2.csv"```

Execute the script.

3. Check Outputs:

The script will display EDA visualizations, model evaluation metrics, and feature importance.

4. Save the Model (Optional):

The trained model will be saved as titanic_model.pkl for later use.

## Model Training and Evaluation

Steps in Model Development:

1. Data Cleaning:

Removed irrelevant columns (e.g., zero, Passengerid, Name, Ticket).

Handled missing values for Age and Embarked.

Scaled numeric features like Fare.

2. Exploratory Data Analysis (EDA):

Visualized relationships between features and survival.

Examined feature distributions.

3. Model Training:

Used Logistic Regression to train the model.

Split the dataset into training (80%) and testing (20%) sets.

4. Evaluation:

Metrics used: Accuracy, Classification Report, ROC-AUC.

Visualized the Confusion Matrix and ROC Curve.

## Key Results
Model Accuracy: ~80% (may vary depending on data splits).

Important Features:

Gender (Sex): Strong predictor of survival.

Passenger Class (Pclass): Lower classes had a higher fatality rate.

Fare: Higher fares correlated with higher survival rates.

ROC-AUC: Curve shows the trade-off between sensitivity and specificity.

## Future Improvements
Use more advanced machine learning models (e.g., Random Forest, XGBoost).

Perform hyperparameter optimization with GridSearchCV.

Add feature engineering, such as extracting titles from passenger names.

Address class imbalance using SMOTE or similar techniques.

Test on additional datasets for robustness.

## Acknowledgments

Dataset Source: Kaggle Titanic Dataset

Tools and Libraries: Python, Pandas, Scikit-learn









