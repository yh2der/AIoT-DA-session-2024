import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which is thread-safe
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import Flask, render_template, request, jsonify
from threading import Lock

app = Flask(__name__)
plot_lock = Lock()

def generate_data(a, num_points, noise):
    x = np.linspace(0, 10, num_points)
    y = a * x + np.random.normal(0, noise, num_points)  # b is implicitly 0 here
    return x, y

def perform_regression(x, y):
    model = LinearRegression()
    x_reshaped = x.reshape(-1, 1)
    model.fit(x_reshaped, y)
    y_pred = model.predict(x_reshaped)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return model.coef_[0], model.intercept_, mse, r2, y_pred

def create_plot(x, y, y_pred, a):
    with plot_lock:
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', label='Data points')
        plt.plot(x, y_pred, color='red', label='Regression line')
        plt.plot(x, a*x, color='green', linestyle='--', label=f'True line (y = {a}x)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.legend()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        a = float(request.form['a'])
        noise = float(request.form['noise'])
        num_points = int(request.form['num_points'])
        
        x, y = generate_data(a, num_points, noise)
        slope, intercept, mse, r2, y_pred = perform_regression(x, y)
        plot_data = create_plot(x, y, y_pred, a)
        
        results = {
            'true_slope': a,
            'estimated_slope': slope,
            'estimated_intercept': intercept,
            'mse': mse,
            'r2': r2,
            'plot': plot_data
        }
        
        return jsonify(results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)