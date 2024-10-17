import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template

# CRISP-DM Step 1: Business Understanding
# Objective: Compare Lasso and RFECV for feature selection in Boston housing price prediction

# CRISP-DM Step 2: Data Understanding
# Load the Boston Housing dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)
X = data.drop('medv', axis=1)
y = data['medv']

# CRISP-DM Step 3: Data Preparation
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Function to train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test, features):
    X_train_subset = X_train[features]
    X_test_subset = X_test[features]
    
    model = LinearRegression()
    model.fit(X_train_subset, y_train)
    
    y_pred = model.predict(X_test_subset)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return rmse, r2

# CRISP-DM Step 4: Modeling
# Lasso Feature Selection with tracking
def lasso_feature_selection(X, y, alpha=0.1):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    importance = pd.Series(abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
    selected_features = []
    results = []
    
    for i in range(1, len(X.columns) + 1):
        features = importance.nlargest(i).index.tolist()
        selected_features.append(features[-1])  # Add the newly selected feature
        rmse, r2 = train_and_evaluate(X, X_test_scaled, y, y_test, features)
        results.append({
            'num_features': i,
            'features': ', '.join(features),
            'rmse': rmse,
            'r2': r2
        })
    
    return pd.DataFrame(results), selected_features

# RFECV Feature Selection with tracking
def rfecv_feature_selection(X, y):
    estimator = LinearRegression()
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X, y)
    importance = pd.Series(selector.ranking_, index=X.columns).sort_values()
    selected_features = []
    results = []
    
    for i in range(1, len(X.columns) + 1):
        features = importance.nsmallest(i).index.tolist()
        selected_features.append(features[-1])  # Add the newly selected feature
        rmse, r2 = train_and_evaluate(X, X_test_scaled, y, y_test, features)
        results.append({
            'num_features': i,
            'features': ', '.join(features),
            'rmse': rmse,
            'r2': r2
        })
    
    return pd.DataFrame(results), selected_features

# Run feature selection methods
lasso_results, lasso_features = lasso_feature_selection(X_train_scaled, y_train)
rfecv_results, rfecv_features = rfecv_feature_selection(X_train_scaled, y_train)

# CRISP-DM Step 5: Evaluation
# The evaluation is performed within the feature selection functions and results are stored in lasso_results and rfecv_results

# CRISP-DM Step 6: Deployment
# Create Flask app for deployment
app = Flask(__name__)

@app.route('/')
def index():
    # Create RMSE vs Number of Features plot
    plt.figure(figsize=(12, 6))
    plt.plot(lasso_results['num_features'], lasso_results['rmse'], marker='o', label='Lasso')
    plt.plot(rfecv_results['num_features'], rfecv_results['rmse'], marker='s', label='RFECV')
    plt.title('RMSE vs Number of Features: Lasso vs RFECV')
    plt.xlabel('Number of Features')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    # Save plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # Prepare data for HTML tables
    lasso_table = lasso_results.to_html(classes='table table-striped', index=False)
    rfecv_table = rfecv_results.to_html(classes='table table-striped', index=False)

    # Prepare final comparison
    lasso_final = lasso_results.iloc[-1]
    rfecv_final = rfecv_results.iloc[-1]
    comparison = pd.DataFrame({
        'Method': ['Lasso', 'RFECV'],
        'RMSE': [lasso_final['rmse'], rfecv_final['rmse']],
        'R2': [lasso_final['r2'], rfecv_final['r2']],
        'Selected Features': [lasso_final['features'], rfecv_final['features']]
    })
    comparison_table = comparison.to_html(classes='table table-striped', index=False)

    return render_template('index.html', 
                           plot_data=plot_data, 
                           lasso_table=lasso_table, 
                           rfecv_table=rfecv_table,
                           comparison_table=comparison_table,
                           lasso_features=", ".join(lasso_features),
                           rfecv_features=", ".join(rfecv_features))

if __name__ == '__main__':
    app.run(debug=True)