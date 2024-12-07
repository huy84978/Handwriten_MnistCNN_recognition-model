import numpy as np
import gzip
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import sys
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


train_images_path = r'C:/python/a/Handwriten_recognition/.venv/data/train-images-idx3-ubyte.gz'
train_labels_path = r'C:/python/a/Handwriten_recognition/.venv/data/train-labels-idx1-ubyte.gz'
test_images_path = r'C:/python/a/Handwriten_recognition/.venv/data/t10k-images-idx3-ubyte.gz'
test_labels_path = r'C:/python/a/Handwriten_recognition/.venv/data/t10k-labels-idx1-ubyte.gz'


def load_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        # Bỏ qua các phần đầu của tệp (16 byte)
        f.read(16)
        # Đọc dữ liệu còn lại và chuyển đổi thành mảng numpy
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8)
        # Định hình lại thành (số lượng ảnh, 28, 28)
        images = images.reshape(-1, 28, 28)
    return images


def load_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        # Bỏ qua các phần đầu của tệp (8 byte)
        f.read(8)
        # Đọc dữ liệu còn lại và chuyển đổi thành mảng numpy
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    return labels


# Tải dữ liệu
train_images = load_images(train_images_path) / 255.0  # Chuẩn hóa dữ liệu
train_labels = load_labels(train_labels_path)
test_images = load_images(test_images_path) / 255.0
test_labels = load_labels(test_labels_path)

# Xây dựng mô hình CNN
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),  # Lớp tích chập 1
    MaxPooling2D(pool_size=(2, 2)),  # Lớp pooling
    Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Lớp tích chập 2
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),  # Lớp fully connected
    Dense(10, activation='softmax')  # Lớp đầu ra
])

# Biên dịch mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
model.save('MnistCNN.keras')


model_loss, model_accuracy = model.evaluate(test_images, test_labels)
print(f"\nmodel Loss: {model_loss}")
print(f"model Accuracy: {model_accuracy}")
