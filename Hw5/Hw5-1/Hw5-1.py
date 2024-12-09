import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import matplotlib.pyplot as plt  # 引入 matplotlib

# 加載 Iris 資料集
iris = load_iris()
X = iris.data  # 特徵
y = iris.target.reshape(-1, 1)  # 標籤

# One-hot 編碼標籤
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定義模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 打印模型架構
print("Model Summary:")
model.summary()

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard 設定
log_dir = "logs/iris"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 訓練模型
history = model.fit(X_train, y_train,
                    epochs=50,
                    validation_data=(X_test, y_test),
                    callbacks=[tensorboard_callback],
                    verbose=1)

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 繪製訓練和驗證損失圖
plt.plot(history.history['loss'], label='Train Loss')  # 訓練損失
plt.plot(history.history['val_loss'], label='Val Loss')  # 驗證損失
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 啟動 TensorBoard 提示
print("\nTo view TensorBoard logs, run the following command in your terminal:")
print(f"tensorboard --logdir={log_dir}")
