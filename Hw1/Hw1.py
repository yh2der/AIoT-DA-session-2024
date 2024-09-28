from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

def generate_data(seed, noise):
    np.random.seed(seed)
    X = np.random.rand(100, 1)
    y = 2 + 3 * X + np.random.randn(100, 1) * noise
    return X, y

def fit_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def create_plot(X, y, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='b', label='Data')
    X_test = np.array([[0], [1]])
    y_pred = model.predict(X_test)
    plt.plot(X_test, y_pred, color='r', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Linear Regression Example')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        seed = int(request.form['seed'])
        noise = float(request.form['noise'])
        
        X, y = generate_data(seed, noise)
        model = fit_model(X, y)
        
        plot_url = create_plot(X, y, model)
        
        return jsonify({
            'intercept': f"{model.intercept_[0]:.2f}",
            'slope': f"{model.coef_[0][0]:.2f}",
            'plot_url': plot_url
        })
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)