from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

# 初始化 Flask 應用
app = Flask(__name__)

# 加載數據
data = pd.read_csv("data/2330-training.csv")
data[['y', 'x1', 'x2', 'x3', 'x4', 'x5']] = data[['y', 'x1', 'x2', 'x3', 'x4', 'x5']].replace({',': ''}, regex=True)
data[['y', 'x1', 'x2', 'x3', 'x4', 'x5']] = data[['y', 'x1', 'x2', 'x3', 'x4', 'x5']].apply(pd.to_numeric)
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# 特徵和目標變量
X = data[['x1', 'x2', 'x3', 'x4', 'x5']]
y = data['y']

# Auto Regression 模型構建
y_series = data['y']
ar_model = AutoReg(y_series, lags=5).fit()
ar_mse = mean_squared_error(y_series[-10:], ar_model.predict(start=len(y_series) - 10, end=len(y_series) - 1, dynamic=False))
ar_r2 = r2_score(y_series[-10:], ar_model.predict(start=len(y_series) - 10, end=len(y_series) - 1, dynamic=False))

# 預測未來 10 個步長
future_steps = 10
future_predictions = ar_model.predict(start=len(y_series), end=len(y_series) + future_steps - 1, dynamic=False)

# 生成 Auto Regression 結果圖表（包含未來趨勢）
ar_fig, ar_ax = plt.subplots()
y_ar_pred = ar_model.predict(start=len(y_series) - 10, end=len(y_series) - 1, dynamic=False)
ar_ax.plot(y_series.index, y_series, label='Actual', color='blue')
ar_ax.plot(y_series[-10:].index, y_ar_pred, label='Predicted', linestyle='--', color='red')

# 添加未來預測的趨勢（使用綠色）
future_index = np.arange(len(y_series), len(y_series) + future_steps)
ar_ax.plot(future_index, future_predictions, label='Future Prediction', linestyle='--', color='green')

ar_ax.set_title('Auto Regression Model Results with Future Predictions')
ar_ax.set_xlabel('Time Index')
ar_ax.set_ylabel('Value')
ar_ax.legend()

# 圖片轉換為字符串格式
ar_buffer = BytesIO()
plt.savefig(ar_buffer, format="png")
ar_buffer.seek(0)
ar_image_png = ar_buffer.getvalue()
ar_buffer.close()
ar_plot_url = base64.b64encode(ar_image_png).decode('utf-8')

# 主頁面，提供多種模型選擇並顯示 Auto Regression 圖表
@app.route('/')
def index():
    return render_template('index.html', feature_columns=X.columns, ar_mse=ar_mse, ar_r2=ar_r2, ar_plot_url=ar_plot_url)

# 處理模型選擇與結果展示
@app.route('/result', methods=['POST'])
def result():
    selected_features = request.form.getlist('features')
    X_subset = X[selected_features]

    # 建立多元線性回歸模型
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 模型評估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 繪製多元線性回歸預測圖表
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Linear Regression with Features: {selected_features}')
    
    # 圖片轉換為字符串格式，嵌入到 HTML 中
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plot_url = base64.b64encode(image_png).decode('utf-8')
    
    return render_template('result.html', mse=mse, r2=r2, plot_url=plot_url, features=selected_features, ar_plot_url=ar_plot_url)

if __name__ == '__main__':
    app.run(debug=True)
