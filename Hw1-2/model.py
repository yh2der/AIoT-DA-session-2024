import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('2330-training.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Clean the data
def clean_numeric_column(df, column):
    df[column] = df[column].replace({',': ''}, regex=True)
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

# Clean all numeric columns
numeric_columns = ['y', 'x1', 'x2', 'x3', 'x4', 'x5']
for col in numeric_columns:
    df = clean_numeric_column(df, col)

# Remove any rows with NaN values
df.dropna(inplace=True)

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Plot the stock price over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['y'])
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

# Auto Regression
def perform_auto_regression(data, lag=5):
    model = AutoReg(data['y'], lags=lag)
    model_fit = model.fit()
    print("Auto Regression Summary:")
    print(model_fit.summary())
    
    # Make predictions
    predictions = model_fit.predict(start=len(data), end=len(data)+10, dynamic=False)
    return predictions

try:
    ar_predictions = perform_auto_regression(df)
    print("Auto Regression Predictions:")
    print(ar_predictions)
except Exception as e:
    print(f"Error in Auto Regression: {str(e)}")

# Multiple Linear Regression
# Feature selection
def select_features(data, target, threshold=0.5):
    corr = data.corr()[target].abs().sort_values(ascending=False)
    selected_features = corr[corr > threshold].index.tolist()
    selected_features.remove(target)
    return selected_features

selected_features = select_features(df, 'y')
print("Selected features:", selected_features)

# Prepare data for multiple linear regression
X = df[selected_features]
y = df['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Multiple Linear Regression Results:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Stock Price")
plt.ylabel("Predicted Stock Price")
plt.title("Actual vs Predicted Stock Prices")
plt.show()

# Save the model for later use in Flask app
import joblib
joblib.dump(model, 'stock_price_model.joblib')