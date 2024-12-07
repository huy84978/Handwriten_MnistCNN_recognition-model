import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Đường dẫn mô hình đã lưu
MODEL_PATH = 'updateMnistCNN.keras'

# Tải mô hình
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Hàm dự đoán ảnh
def predict_image(image_path):
    try:
        # Mở ảnh và chuyển sang grayscale
        img = Image.open(image_path).convert('L')  # Chuyển sang ảnh đen trắng
        img = ImageOps.invert(img)  # Đảo ngược màu nếu cần (nếu nền đen)
        img = img.resize((28, 28))  # Resize về kích thước 28x28
        img_array = np.array(img) / 255.0  # Chuẩn hóa giá trị pixel (0-1)
        img_array = img_array.reshape(1, 28, 28, 1)  # Thêm chiều batch

        # Dự đoán
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)  # Lấy nhãn có xác suất cao nhất
        return predicted_label, prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Hàm mở file và hiển thị dự đoán
def open_and_predict():
    # Mở hộp thoại chọn file
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not file_path:
        return  # Người dùng hủy chọn file

    # Dự đoán ảnh
    predicted_label, prediction = predict_image(file_path)
    if predicted_label is not None:
        messagebox.showinfo("Prediction Result", f"Predicted Label: {predicted_label}")
    else:
        messagebox.showerror("Error", "Failed to predict the image.")

# Giao diện Tkinter
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# Nhãn
label = tk.Label(root, text="Handwritten Digit Recognition", font=("Arial", 16))
label.pack(pady=10)

# Nút mở ảnh
open_button = tk.Button(root, text="Open Image", command=open_and_predict, font=("Arial", 14), bg="lightblue")
open_button.pack(pady=10)

# Khởi chạy ứng dụng
root.mainloop()
