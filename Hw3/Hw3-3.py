import streamlit as st
import numpy as np
from sklearn.svm import SVC
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 設置頁面配置
st.set_page_config(layout="wide")

def generate_complex_data(n_samples=300, noise_level=0.3, pattern_type='moon'):
    np.random.seed(42)
    
    if pattern_type == 'moon':
        # 生成月牙形數據
        t = np.random.uniform(0, np.pi, n_samples//2)
        # 第一個月牙
        X1 = np.column_stack([
            3 * np.cos(t),
            3 * np.sin(t) + noise_level * np.random.randn(n_samples//2)
        ])
        # 第二個月牙
        X2 = np.column_stack([
            2.5 * np.cos(t + np.pi),
            2.5 * np.sin(t + np.pi) + 2 + noise_level * np.random.randn(n_samples//2)
        ])
    
    elif pattern_type == 'spiral':
        # 生成螺旋形數據
        t = np.linspace(0, 2*np.pi, n_samples//2)
        r = t + 1
        # 第一個螺旋
        X1 = np.column_stack([
            r * np.cos(t) + noise_level * np.random.randn(n_samples//2),
            r * np.sin(t) + noise_level * np.random.randn(n_samples//2)
        ])
        # 第二個螺旋
        X2 = np.column_stack([
            r * np.cos(t + np.pi) + noise_level * np.random.randn(n_samples//2),
            r * np.sin(t + np.pi) + noise_level * np.random.randn(n_samples//2)
        ])
    
    else:  # S形
        # 生成S形數據
        t = np.linspace(0, 4*np.pi, n_samples//2)
        X1 = np.column_stack([
            np.sin(t) + noise_level * np.random.randn(n_samples//2),
            t/2 + noise_level * np.random.randn(n_samples//2)
        ])
        X2 = np.column_stack([
            np.sin(t + np.pi) + 2 + noise_level * np.random.randn(n_samples//2),
            t/2 + noise_level * np.random.randn(n_samples//2)
        ])

    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # 標準化數據
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# 標題
st.title('2D SVM with Complex Decision Boundary')

# 數據參數
col1, col2, col3, col4 = st.columns(4)
with col1:
    pattern = st.selectbox('Pattern Type', ['moon', 'spiral', 'S-shape'])
with col2:
    noise = st.slider('Noise Level', 0.1, 1.0, 0.3)
with col3:
    test_size = st.slider('Test Set Size', 0.1, 0.5, 0.2)
with col4:
    n_samples = st.slider('Sample Size', 100, 1000, 300)

# 生成數據
X, y = generate_complex_data(n_samples=n_samples, noise_level=noise, pattern_type=pattern)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 模型參數設置
col1, col2 = st.columns(2)
with col1:
    C = st.slider('Regularization (C)', 0.1, 20.0, 1.0)
with col2:
    gamma = st.slider('RBF Kernel (gamma)', 0.1, 5.0, 0.5)

# 訓練模型
model = SVC(kernel='rbf', C=C, gamma=gamma)
model.fit(X_train, y_train)

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
    
    # 訓練數據點
    go.Scatter3d(x=X_train[y_train==0, 0], y=X_train[y_train==0, 1], 
                 z=np.zeros_like(X_train[y_train==0, 0]),
                 mode='markers', marker=dict(size=5, color='blue'),
                 name='Train Class 0'),
    go.Scatter3d(x=X_train[y_train==1, 0], y=X_train[y_train==1, 1], 
                 z=np.zeros_like(X_train[y_train==1, 0]),
                 mode='markers', marker=dict(size=5, color='red'),
                 name='Train Class 1'),
    
    # 測試數據點
    go.Scatter3d(x=X_test[y_test==0, 0], y=X_test[y_test==0, 1], 
                 z=np.zeros_like(X_test[y_test==0, 0]),
                 mode='markers', marker=dict(size=5, color='lightblue', symbol='x'),
                 name='Test Class 0'),
    go.Scatter3d(x=X_test[y_test==1, 0], y=X_test[y_test==1, 1], 
                 z=np.zeros_like(X_test[y_test==1, 0]),
                 mode='markers', marker=dict(size=5, color='pink', symbol='x'),
                 name='Test Class 1')
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

# 計算並顯示訓練集和測試集的準確率
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

col1, col2 = st.columns(2)
with col1:
    st.write(f'Training Accuracy: {train_accuracy:.3f}')
with col2:
    st.write(f'Testing Accuracy: {test_accuracy:.3f}')

# 添加模型評估信息
if train_accuracy - test_accuracy > 0.15:
    st.warning('警告：模型可能存在過擬合現象！')
    st.info('建議：\n'
            '1. 減小gamma值\n'
            '2. 減小C值\n'
            '3. 增加訓練數據量')
elif train_accuracy < 0.8:
    st.warning('警告：模型可能存在欠擬合現象！')
    st.info('建議：\n'
            '1. 增加gamma值\n'
            '2. 增加C值\n'
            '3. 使用更複雜的核函數')