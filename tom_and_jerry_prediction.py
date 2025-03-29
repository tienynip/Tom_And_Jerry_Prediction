import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from sklearn.svm import SVC
import joblib

data_dir = "tom_and_jerry"
categories = ['jerry','tom']

img_size = 50 #Đặt kích thước hình ảnh
X = [] # Lưu trữ ảnh dưới dạng vector
y = [] # Nhãn của ảnh

for label, category in enumerate(categories): #Duyệt và từng thư mục và gán nhãn vào categories label = 0, category = 'jerry', label = 1, category = 'tom'
    folder_path = os.path.join(data_dir,category) # Tạo đường dẫn tới thư mục ảnh
    for file_name in os.listdir(folder_path): #Duyệt qua từng danh sách trong mục
        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img,(img_size,img_size))
        img = img.flatten() #Chuyển ảnh thành vector
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác: {accuracy:.2f}")
# joblib.dump(model, "model.pkl")


"""Test bằng cách nhập bàn phím"""
# def process_image(image_path, img_size=50):
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Không thể đọc ảnh!")
#     # Resize ảnh về đúng kích thước
#     img = cv2.resize(img, (img_size, img_size))

#     # Chuyển ảnh thành vector 1D
#     img = img.flatten()

#     # Chuyển thành ma trận 2D (1 ảnh)
#     img = np.array(img).reshape(1, -1)  # Định dạng phù hợp cho model.predict()

#     return img  # Trả về dữ liệu đã xử lý

# image_path = "jerry.jpg"
# processed_img = process_image(image_path)
# model.predict(processed_img)[0]  # Truyền vào dữ liệu đã xử lý

# print("Đây là:", categories[model.predict(processed_img)[0]])
