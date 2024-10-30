import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def get_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    print(f"訓練集形狀: {train_data.shape}")
    print(f"測試集形狀: {test_data.shape}")
    return train_data, test_data

def preprocess_data(df):
    df = df.copy()
    
    # 1. 基本缺失值處理
    df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    
    # 2. 增強特徵工程
    # 2.1 名字特徵
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,
        'Dr': 5, 'Rev': 5, 'Major': 5, 'Col': 5,
        'Mlle': 2, 'Ms': 2, 'Lady': 3, 'Sir': 1,
        'Mme': 3, 'Don': 1, 'Countess': 3, 'Jonkheer': 1,
        'Capt': 5
    }
    df['Title'] = df['Title'].map(title_mapping)
    
    # 2.2 家庭特徵
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FamilyType'] = pd.cut(df['FamilySize'], 
                             bins=[0, 1, 4, float('inf')],
                             labels=[0, 1, 2])  # 獨居、小家庭、大家庭
    
    # 2.3 客艙等級相關特徵
    df['Pclass_Fare'] = df['Pclass'] * df['Fare']
    
    # 2.4 年齡相關特徵
    df['AgeBin'] = pd.cut(df['Age'], 
                         bins=[0, 12, 20, 30, 50, float('inf')],
                         labels=[0, 1, 2, 3, 4])
    df['Age*Class'] = df['Age'] * df['Pclass']
    
    # 2.5 票價分箱
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3])
    
    # 2.6 Cabin相關特徵
    df['HasCabin'] = (~df['Cabin'].isna()).astype(int)
    
    # 3. 類別變量轉換
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # 4. 選擇最終特徵
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'FamilySize', 
               'IsAlone', 'FamilyType', 'Pclass_Fare', 'AgeBin', 
               'Age*Class', 'FareBin', 'HasCabin', 'Embarked']
    
    return df[features]

def train_model(X_train, y_train):
    # 使用 GridSearchCV 尋找最佳參數
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                             cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print("最佳參數:", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, X_train, y_train, feature_names):
    # 訓練集性能
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # 測試集性能
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    
    # 交叉驗證得分
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    print("\n模型評估報告:")
    print(f"訓練集準確率: {train_accuracy:.4f}")
    print(f"測試集準確率: {test_accuracy:.4f}")
    print(f"交叉驗證準確率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("\n分類報告:")
    print(classification_report(y_test, y_pred))
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n特徵重要性:")
    print(feature_importance)
    
    # 繪製特徵重要性圖
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance')
    plt.show()
    
    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return test_accuracy, cm

def main():
    # 獲取數據
    train_data, test_data = get_data()
    
    # 準備特徵和目標變量
    X = preprocess_data(train_data)
    y = train_data['Survived']
    
    # 保存特徵名稱
    feature_names = X.columns
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # 特徵縮放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 訓練優化後的模型
    model = train_model(X_train_scaled, y_train)
    
    # 評估模型
    accuracy, confusion_mat = evaluate_model(
        model, X_test_scaled, y_test, 
        X_train_scaled, y_train, 
        feature_names  # 傳入特徵名稱
    )
    
    # 處理測試集並生成提交文件
    test_features = preprocess_data(test_data)
    test_features_scaled = scaler.transform(test_features)
    predictions = model.predict(test_features_scaled)
    
    submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created successfully!")
    
if __name__ == "__main__":
    main()