import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import *

# Load data
data_train = pd.read_csv('./data/train.csv')
data_test = pd.read_csv('./data/test.csv')

# remove PassengerId
train_data, test_data = train_test_split(
    data_train.drop(['PassengerId'], axis=1), 
    random_state=100, 
    train_size=0.8
)

# Set up enviorment of PyCaret
clf1 = setup(
    data = train_data,
    target = 'Survived',
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch'],
    categorical_features = ['Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked'],
    ignore_features = ['Name'],  # Ignore Name
    session_id = 123
)

print(models())

# comparison of all model
best_model = compare_models(
    fold=5, 
    include=[
        'lr',      # Logistic Regression
        'knn',     # K Neighbors Classifier
        'nb',      # Naive Bayes
        'dt',      # Decision Tree Classifier
        'svm',     # SVM - Linear Kernel
        'rbfsvm',  # SVM - Radial Kernel
        'gpc',     # Gaussian Process Classifier
        'mlp',     # MLP Classifier
        'ridge',   # Ridge Classifier
        'rf',      # Random Forest Classifier
        'qda',     # Quadratic Discriminant Analysis
        'ada',     # Ada Boost Classifier
        'gbc',     # Gradient Boosting Classifier
        'lda',     # Linear Discriminant Analysis
        'et',      # Extra Trees Classifier
        'lightgbm',# Light Gradient Boosting Machine
    ]
)

predictions = predict_model(best_model, data=data_test)
print(predictions)