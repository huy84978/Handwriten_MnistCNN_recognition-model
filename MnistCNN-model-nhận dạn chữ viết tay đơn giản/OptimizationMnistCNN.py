import numpy as np
import gzip
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Đường dẫn mới của dataset
train_images_path = r'C:/python/a/Handwriten_recognition/.venv/data/train-images-idx3-ubyte.gz'
train_labels_path = r'C:/python/a/Handwriten_recognition/.venv/data/train-labels-idx1-ubyte.gz'
test_images_path = r'C:/python/a/Handwriten_recognition/.venv/data/t10k-images-idx3-ubyte.gz'
test_labels_path = r'C:/python/a/Handwriten_recognition/.venv/data/t10k-labels-idx1-ubyte.gz'

def load_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        f.read(16)
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8)
        images = images.reshape(-1, 28, 28)
    return images

def load_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        f.read(8)
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    return labels

# Tải dữ liệu
train_images = load_images(train_images_path) / 255.0
train_labels = load_labels(train_labels_path)
test_images = load_images(test_images_path) / 255.0
test_labels = load_labels(test_labels_path)

# Thêm Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    zoom_range=0.1  
    
)
datagen.fit(train_images.reshape(-1, 28, 28, 1))

# Xây dựng mô hình
model = Sequential([
    Input(shape=(28, 28, 1)),  

    Conv2D(64, kernel_size=(3, 3), activation='relu'),  
    BatchNormalization(),  
    MaxPooling2D(pool_size=(2, 2)),  
    Dropout(0.25),  

    Conv2D(128, kernel_size=(3, 3), activation='relu'),  
    BatchNormalization(),  
    MaxPooling2D(pool_size=(2, 2)),  
    Dropout(0.25),  

    Conv2D(256, kernel_size=(3, 3), activation='relu'),  
    BatchNormalization(),  
    Flatten(),  

    Dense(256, activation='relu'),  
    Dropout(0.5),  
    Dense(10, activation='softmax')  
])

# Biên dịch mô hình
model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])  

# **THAY ĐỔI: Thêm EarlyStopping**
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) 

# Huấn luyện mô hình
model.fit(
    datagen.flow(train_images.reshape(-1, 28, 28, 1), train_labels, batch_size=64),  
    epochs=40,  
    validation_data=(test_images.reshape(-1, 28, 28, 1), test_labels),  
    callbacks=[early_stopping]  
)

# Lưu mô hình
model.save('updateMnistCNN.keras')  

# Đánh giá mô hình
model_loss, model_accuracy = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print(f"\nModel Loss: {model_loss}")
print(f"Model Accuracy: {model_accuracy}")
