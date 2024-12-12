import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data to range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data for Dense NN (flattening images)
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# Reshape data for CNN (adding channel dimension)
x_train_cnn = x_train[..., np.newaxis]
x_test_cnn = x_test[..., np.newaxis]

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Dense NN Model
dense_model = Sequential([
    Dense(128, activation='relu', input_shape=(28*28,), kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Compile the model
dense_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add callbacks for early stopping and learning rate adjustment
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train the model
dense_history = dense_model.fit(
    x_train_flat, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test_flat, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# CNN Model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
cnn_history = cnn_model.fit(
    x_train_cnn, y_train,
    epochs=10, 
    batch_size=32,
    validation_data=(x_test_cnn, y_test)
)

# Evaluate Dense NN
dense_test_loss, dense_test_acc = dense_model.evaluate(x_test_flat, y_test, verbose=0)
print(f"Dense NN Test Accuracy: {dense_test_acc:.4f}")

# Evaluate CNN
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)
print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")

# Plot training & validation accuracy for Dense NN
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(dense_history.history['accuracy'], label='Train Acc')
plt.plot(dense_history.history['val_accuracy'], label='Val Acc')
plt.title('Dense NN Accuracy')
plt.legend()

# Plot training & validation accuracy for CNN
plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['accuracy'], label='Train Acc')
plt.plot(cnn_history.history['val_accuracy'], label='Val Acc')
plt.title('CNN Accuracy')
plt.legend()

plt.show()