import streamlit as st
import numpy as np
from sklearn.svm import SVC
import plotly.graph_objects as go

# 設置頁面配置
st.set_page_config(layout="wide")

def generate_circular_data(n_samples=300):
    np.random.seed(42)
    
    # 內圈數據（類別0）
    r_inner = np.random.normal(2, 0.3, n_samples//2)
    theta_inner = np.random.uniform(0, 2*np.pi, n_samples//2)
    X_inner = np.column_stack([
        r_inner * np.cos(theta_inner),
        r_inner * np.sin(theta_inner)
    ])
    
    # 外圈數據（類別1）
    r_outer = np.random.normal(4, 0.3, n_samples//2)
    theta_outer = np.random.uniform(0, 2*np.pi, n_samples//2)
    X_outer = np.column_stack([
        r_outer * np.cos(theta_outer),
        r_outer * np.sin(theta_outer)
    ])
    
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    return X, y

# 標題
st.title('2D SVM with 3D Decision Boundary')

# 生成數據
X, y = generate_circular_data()

# 模型參數設置
col1, col2 = st.columns(2)
with col1:
    C = st.slider('Regularization (C)', 0.1, 10.0, 1.0)
with col2:
    gamma = st.slider('RBF Kernel (gamma)', 0.1, 2.0, 0.5)

# 訓練模型
model = SVC(kernel='rbf', C=C, gamma=gamma)
model.fit(X, y)

# 創建決策邊界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                    np.linspace(y_min, y_max, 50))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 創建3D圖
fig = go.Figure(data=[
    # 決策平面
    go.Surface(x=xx, y=yy, z=Z, colorscale='RdBu', opacity=0.7),
    
    # 數據點
    go.Scatter3d(x=X[y==0, 0], y=X[y==0, 1], z=np.zeros_like(X[y==0, 0]),
                 mode='markers', marker=dict(size=5, color='blue'),
                 name='Class 0'),
    go.Scatter3d(x=X[y==1, 0], y=X[y==1, 1], z=np.zeros_like(X[y==1, 0]),
                 mode='markers', marker=dict(size=5, color='red'),
                 name='Class 1')
])

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Decision Function'
    ),
    width=800,
    height=800
)

# 顯示圖形和準確率
st.plotly_chart(fig)
accuracy = model.score(X, y)
st.write(f'Model Accuracy: {accuracy:.2f}')