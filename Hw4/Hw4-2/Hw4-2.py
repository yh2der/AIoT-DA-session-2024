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

# 處理缺失值，並創造新的特徵
data_train['Age'].fillna(data_train['Age'].median(), inplace=True)
data_train['Fare'].fillna(data_train['Fare'].median(), inplace=True)
data_train['Embarked'].fillna(data_train['Embarked'].mode()[0], inplace=True)
data_train['Cabin'].fillna('Unknown', inplace=True)

# 創造新特徵
data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch'] + 1
data_train['IsAlone'] = (data_train['FamilySize'] == 1).astype(int)
data_train['Title'] = data_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data_train['FareBin'] = pd.qcut(data_train['Fare'], 4, labels=False)

# 處理測試集中的相同缺失值和新特徵
data_test['Age'].fillna(data_test['Age'].median(), inplace=True)
data_test['Fare'].fillna(data_test['Fare'].median(), inplace=True)
data_test['Embarked'].fillna(data_test['Embarked'].mode()[0], inplace=True)
data_test['Cabin'].fillna('Unknown', inplace=True)

data_test['FamilySize'] = data_test['SibSp'] + data_test['Parch'] + 1
data_test['IsAlone'] = (data_test['FamilySize'] == 1).astype(int)
data_test['Title'] = data_test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data_test['FareBin'] = pd.qcut(data_test['Fare'], 4, labels=False)

# 移除無用特徵
train_data = data_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_data = data_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# 使用 PyCaret 進行 setup
clf1 = setup(
    data=train_data,
    target='Survived',
    numeric_features=['Age', 'Fare', 'FamilySize', 'FareBin'],
    categorical_features=['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'IsAlone'],
    session_id=123
)

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
    ],
    sort='Accuracy'
)

# 模型集成（Stacking）
ensemble_model = blend_models(
    estimator_list=[best_model, create_model('rf'), create_model('gbc')],
    fold=5
)

# 使用 PyCaret 的 tune_model 進行超參數搜索
optimized_model = tune_model(
    ensemble_model,
    fold=5,
    optimize='Accuracy',
    search_library='optuna',  # 使用 Optuna 作為超參數優化工具
    search_algorithm='random'
)

# 使用最佳模型進行預測
predictions = predict_model(optimized_model, data=test_data)
print(predictions)