import os
import cv2
import random
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


data_dir = "/content/drive/MyDrive/Data/train"
if not os.path.isdir(data_dir):
    print(f"Lỗi: {data_dir} không phải là một thư mục hợp lệ!")
else:
    print(f"{data_dir} là một thư mục hợp lệ.")

# Lấy danh sách các lớp
categories = os.listdir(data_dir)

# Thống kê số lượng ảnh trong mỗi lớp
original_counts = {}
for category in categories:
    class_path = os.path.join(data_dir, category)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    original_counts[category] = len(image_files)

# Vẽ biểu đồ số lượng ảnh gốc
plt.figure(figsize=(10, 6))
bars = plt.bar(original_counts.keys(), original_counts.values(), color='salmon')
plt.title("Số lượng ảnh gốc trong mỗi lớp")
plt.xlabel("Tên lớp (Class)")
plt.ylabel("Số ảnh gốc")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ghi số lượng lên cột
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, str(height),
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

def adjust_brightness(image):
    factor = random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def rotate_image(image):
    angle = random.uniform(-20, 20)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def flip_image(image):
    flip_mode = random.choice([-1, 0, 1])
    return cv2.flip(image, flip_mode)

def shear_image(image):
    shear_factor = random.uniform(-0.2, 0.2)
    h, w = image.shape[:2]
    M = np.array([[1, shear_factor, 0], [shear_factor, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(image, M, (w, h))



X, y = [], []
showed = False  # Cờ để chỉ hiển thị ảnh một lần

# Augmentation function như cũ
def augment_image(img, n_augments=3):
    images = [img]
    for _ in range(n_augments):
        aug_img = img.copy()
        if random.random() < 0.5: aug_img = rotate_image(aug_img)
        if random.random() < 0.5: aug_img = flip_image(aug_img)
        if random.random() < 0.5: aug_img = shear_image(aug_img)
        if random.random() < 0.5: aug_img = adjust_brightness(aug_img)
        images.append(aug_img)
    return images

# Trích xuất HOG
for idx, category in enumerate(categories):
    class_path = os.path.join(data_dir, category)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))

        augmented_images = augment_image(img, n_augments=3)

        # Hiển thị 4 ảnh augment đầu tiên (chỉ 1 lần)
        if not showed:
            titles = ['Ảnh gốc', 'Augment 1', 'Augment 2', 'Augment 3']
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
            for i in range(4):
                img_rgb = cv2.cvtColor(augmented_images[i], cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(titles[i])
                axes[i].axis('off')
            plt.suptitle(f"Hiển thị 4 ảnh augment - lớp: {category}")
            plt.tight_layout()
            plt.show()

        for i, aug_img in enumerate(augmented_images):
            gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
            features, hog_image = hog(
                gray,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                feature_vector=True,
                visualize=True
            )
            X.append(features)
            y.append(idx)

            # Hiển thị 1 ảnh gốc + HOG (chỉ 1 lần)
            if not showed:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                axes[0].imshow(img_rgb)
                axes[0].set_title("Ảnh augment bất kỳ")
                axes[0].axis("off")

                axes[1].imshow(hog_image, cmap='gray')
                axes[1].set_title("HOG features")
                axes[1].axis("off")

                plt.suptitle(f"Ảnh và đặc trưng HOG - lớp: {category}")
                plt.tight_layout()
                plt.show()

                showed = True  # Cờ này sẽ ngăn lặp lại phần hiển thị

# Chuyển về numpy
X = np.array(X)
y = np.array(y)
print("HOG features trích xuất xong:", X.shape)

# Chuẩn hóa và PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)  # Giữ lại 95% phương sai
X_pca = pca.fit_transform(X_scaled)

# Lưu kết quả
np.savez_compressed('/content/drive/MyDrive/X_pca.npz', X_pca=X_pca)
np.savez_compressed('/content/drive/MyDrive/y.npz', y=y)

# Lưu scaler và PCA để dùng lại
joblib.dump(scaler, '/content/drive/MyDrive/scaler.pkl')
joblib.dump(pca, '/content/drive/MyDrive/pca.pkl')

print(f"Đã lưu features PCA với shape: {X_pca.shape}, labels: {y.shape}")

import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Load dữ liệu đã PCA
X = np.load('/content/drive/MyDrive/X_pca.npy')  # Đã PCA rồi
y = np.load('/content/drive/MyDrive/y.npy')

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f" Dataset đã PCA có {len(X)} mẫu, số đặc trưng mỗi ảnh: {X.shape[1]}")

# Lưu train/test set
np.save('/content/drive/MyDrive/X_train.npy', X_train)
np.save('/content/drive/MyDrive/X_test.npy', X_test)
np.save('/content/drive/MyDrive/y_train.npy', y_train)
np.save('/content/drive/MyDrive/y_test.npy', y_test)

print(" Đã lưu tập train/test sau PCA thành công.")



# Load dữ liệu đã xử lý
X_train = np.load('/content/drive/MyDrive/X_train.npy')
X_test = np.load('/content/drive/MyDrive/X_test.npy')
y_train = np.load('/content/drive/MyDrive/y_train.npy')
y_test = np.load('/content/drive/MyDrive/y_test.npy')

# Load scaler và PCA nếu cần dùng tiếp
scaler_path = '/content/drive/MyDrive/scaler(1).pkl'
pca_path = '/content/drive/MyDrive/pca(1).pkl'

scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)


#categories = ['class_1', 'class_2', 'class_3']  

# ==============================
print("Bắt đầu train mô hình SVM...")

# Khởi tạo mô hình SVM
svm_model = SVC(kernel="rbf", C=10, gamma="scale")

# Train model
svm_model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Độ chính xác trên tập test: {accuracy * 100:.2f}%")

# Báo cáo chi tiết
print("\n Báo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=categories))

# Vẽ ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Ma trận nhầm lẫn")
plt.tight_layout()
plt.show()

# ==============================
# Lưu model, scaler và PCA sau khi training xong
model_path = '/content/drive/MyDrive/svm_model(1).pkl'

joblib.dump(svm_model, model_path)
joblib.dump(scaler, scaler_path)  # đã load ở trên
joblib.dump(pca, pca_path)        # đã load ở trên

print(" Đã lưu mô hình, scaler và PCA thành công!")


def predict_image(image_path, model, scaler):
    # Đọc và xử lý ảnh
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Trích xuất HOG
    hog_feature = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)

    # Chuẩn hóa
    hog_feature = scaler.transform([hog_feature])

    # Dự đoán
    prediction = model.predict(hog_feature)
    return prediction[0]

# Load model và scaler
svm_model = joblib.load("/content/drive/MyDrive/svm_model.pkl")
scaler = joblib.load("/content/drive/MyDrive/scaler.pkl")

# Test với một ảnh mới
image_path = "/content/drive/MyDrive/img/img1.png"
label = predict_image(image_path, svm_model, scaler)
print(f"Ảnh này thuộc lớp: {categories[label]}")
