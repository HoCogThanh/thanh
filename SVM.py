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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from collections import Counter
import seaborn as sns
from sklearn.decomposition import IncrementalPCA

data_dir = "/content/drive/MyDrive/DATASET/Lua"
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

# In ra tổng số ảnh
total_images = sum(original_counts.values())
print(f"Tổng số lượng ảnh trong toàn bộ dữ liệu: {total_images}")

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

# augment_image
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
#Histogram color
def extract_color_histogram(image, bins=(16, 16, 16)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


#Augment_image va trich xuat hog&giam chieu PCA
# ==== Cấu hình đường dẫn và thông tin ====
data_dir = "/content/drive/MyDrive/DATASET/Lua"
dest_dir = '/content/drive/MyDrive/Train_Lua_5'
processed_log_path = "/content/drive/MyDrive/Train_Lua_5/processed_images_lua.txt"

batch_size = 2000
X_batch, y_batch = [], []

# ==== Danh sách lớp ====
categories = [
    'sheath_blight', 'bacterial_leaf_blight',
    'leaf_scald', 'rice_hispa', 'brown_spot'
]

# ==== Đảm bảo thư mục tồn tại ====
os.makedirs(dest_dir, exist_ok=True)

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

# ==== Hiển thị ảnh mẫu ====
random_category = random.choice(categories)
random_class_path = os.path.join(data_dir, random_category)
random_image_name = random.choice(os.listdir(random_class_path))
random_image_path = os.path.join(random_class_path, random_image_name)

sample_img = cv2.imread(random_image_path)
if sample_img is not None:
    sample_img = cv2.resize(sample_img, (128, 128))
    sample_augmented_images = augment_image(sample_img)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ảnh Gốc")
    axes[0].axis('off')

    for i, aug_img in enumerate(sample_augmented_images[1:4]):
        axes[i+1].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        axes[i+1].set_title(f"Biến Đổi {i+1}")
        axes[i+1].axis('off')

    plt.suptitle(f"Hiển thị ngẫu nhiên từ lớp: {random_category}")
    plt.tight_layout()
    plt.show()

# ==== Kiểm tra batch đã tồn tại ====
batch_files = sorted([
    f for f in os.listdir(dest_dir)
    if f.startswith("raw_hog_batch_") and f.endswith(".npz")
])
existing_batches = sorted([
    int(f.split("_")[-1].split(".")[0])
    for f in batch_files
])

# ==== Nếu đủ batch thì xử lý PCA ====
if len(batch_files) == 18:
    print(f"\n Đã có đủ {len(batch_files)} batch. Bắt đầu xử lý PCA và chia tập dl...")

    X_all = np.concatenate([
        np.load(os.path.join(dest_dir, f))['X'] for f in batch_files
    ], axis=0)
    y_all = np.concatenate([
        np.load(os.path.join(dest_dir, f))['y'] for f in batch_files
    ], axis=0)

    print("\n Đang chuẩn hóa và giảm chiều...")
    #Khởi tạo scaler
    scaler = StandardScaler()
    #Khởi tạo IncrementalPCA với số lượng thành phần đã tính
    pca = IncrementalPCA(n_components=500)
    #Lặp qua từng Batch để fit scaler và PCA
    batch_sizePCA = 2000
    n_batchesPCA = len(X_all) // batch_sizePCA
    #scaler và PCA từng batch
    for i in range(n_batchesPCA):
        X_batchPCA = X_all[i * batch_sizePCA:(i + 1) * batch_sizePCA]
        X_scaled = scaler.fit_transform(X_batchPCA) if i == 0 else scaler.transform(X_batchPCA)
        pca.partial_fit(X_scaled)

    joblib.dump(scaler, f"{dest_dir}/scaler.pkl")
    joblib.dump(pca, f"{dest_dir}/pca.pkl")

    X_scaled = scaler.transform(X_all)
    X_pca = pca.transform(X_scaled)


    X_train, X_temp, y_train, y_temp = train_test_split(X_pca, y_all, test_size=0.2, stratify=y_all)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

    np.save(f"{dest_dir}/X_train.npy", X_train)
    np.save(f"{dest_dir}/X_val.npy", X_val)
    np.save(f"{dest_dir}/X_test.npy", X_test)
    np.save(f"{dest_dir}/y_train.npy", y_train)
    np.save(f"{dest_dir}/y_val.npy", y_val)
    np.save(f"{dest_dir}/y_test.npy", y_test)

    print("\nĐã chia và lưu train/val/test sau khi giảm chiều PCA.")

# ==== Nếu chưa đủ batch thì tiếp tục trích xuất ====
else:
    print(f"\n Mới có {len(batch_files)} batch. Cần 18 batch để xử lý PCA.")
    if existing_batches:
        last_batch = max(existing_batches)
        batch_counter = last_batch + 1
        samples_done = len(existing_batches) * batch_size
        print(f" Tiếp tục từ batch {batch_counter} (đã có {len(existing_batches)} batch)")
    else:
        batch_counter = 0
        samples_done = 0

    samples_processed = 0

    if os.path.exists(processed_log_path):
        with open(processed_log_path, "r") as f:
            processed_images = set(line.strip() for line in f)
    else:
        processed_images = set()

    for idx, category in enumerate(categories):
        class_path = os.path.join(data_dir, category)
        for img_name in os.listdir(class_path):
            image_id = f"{category}/{img_name}"
            if image_id in processed_images:
                continue

            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (128, 128))
            augmented_images = augment_image(img)

            if samples_processed < samples_done:
                samples_processed += len(augmented_images)
                continue

            for aug_img in augmented_images:
                gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
                hog_features, _ = hog(
                    gray, pixels_per_cell=(16, 16),
                    cells_per_block=(4, 4),
                    feature_vector=True,
                    visualize=True
                )
                #color_hist = extract_color_histogram(aug_img)
                #combined_features = np.concatenate((hog_features, color_hist))
                X_batch.append(hog_features)
                y_batch.append(idx)

                if len(X_batch) >= batch_size:
                    np.savez_compressed(f"{dest_dir}/raw_hog_batch_{batch_counter}.npz", X=X_batch, y=y_batch)
                    print(f" Đã lưu raw batch {batch_counter} với {len(X_batch)} mẫu")
                    X_batch, y_batch = [], []
                    batch_counter += 1


            processed_images.add(image_id)
            with open(processed_log_path, "a") as f:
                f.write(image_id + "\n")

    if X_batch:
        np.savez_compressed(f"{dest_dir}/raw_hog_batch_{batch_counter}.npz", X=X_batch, y=y_batch)
        print(f" Đã lưu raw batch cuối {batch_counter} với {len(X_batch)} mẫu")
#HSVM & Ramdom forest


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import learning_curve

# ====== Cấu hình đường dẫn ======
DATA_DIR = '/content/drive/MyDrive/Train_Lua_5'

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))

y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

categories = [
    'sheath_blight',
    'bacterial_leaf_blight',
    'leaf_scald',
    'rice_hispa',
    'brown_spot'
]

# ====== Tổng số lượng mẫu ======
print(f"\nSố lượng mẫu:")
print(f"- Train: {len(X_train)} mẫu")
print(f"- Validation: {len(X_val)} mẫu")
print(f"- Test: {len(X_test)} mẫu")

# ====== In ra số lượng mẫu theo từng lớp ======
def print_class_distribution(y, split_name):
    print(f"\nSố lượng mẫu theo từng lớp trong tập {split_name}:")
    counter = Counter(y)
    for label, count in counter.items():
        class_name = categories[label] if label < len(categories) else f"Lớp {label}"
        print(f"  {class_name} ({label}): {count} mẫu")

print_class_distribution(y_train, "Train")
print_class_distribution(y_val, "Validation")
print_class_distribution(y_test, "Test")

'''
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Thử nghiệm các giá trị khác nhau
param_grid = {
    'C': [0.1, 1, 10, 20],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf']
}

# Khởi tạo GridSearchCV với 5-fold cross-validation
svm_model = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)

# Huấn luyện trên tập train
svm_model.fit(X_train, y_train)

# In ra tham số tốt nhất
print("Best parameters found:", svm_model.best_params_)
'''
print("\nĐang huấn luyện mô hình SVM...")
svm_model = SVC(kernel='rbf', C=20.0, gamma='scale')
svm_model.fit(X_train, y_train)

# ====== Đánh giá trên tập validation ======
y_val_pred = svm_model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')
print(f"\n[SVM] Accuracy (val): {val_acc*100:.2f}%")
print(f"[SVM] F1-score (val): {val_f1:.4f}")

# ====== Đánh giá trên tập test ======
y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n[SVM] Accuracy (test): {acc*100:.2f}%")
print(f"[SVM] F1-score (test): {f1:.4f}")
print("\n[SVM] Báo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=categories))

# ====== Lưu mô hình SVM nếu tốt ======
if f1 >= 0.9:
    model_path = os.path.join(DATA_DIR, 'svm_model.pkl')
    joblib.dump(svm_model, model_path)
    print(f"\nĐã lưu mô hình SVM với F1-score: {f1:.4f}")
else:
    print(f"\nF1-score ({f1:.4f}) không đủ ngưỡng để lưu mô hình.")

# ====== Vẽ ma trận nhầm lẫn cho SVM ======
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Ma trận nhầm lẫn - SVM")
plt.tight_layout()
plt.show()

# ====== Precision theo lớp (SVM) ======
report = classification_report(y_test, y_pred, target_names=categories, output_dict=True)
precision_values = [report[cat]['precision'] for cat in categories]

plt.figure(figsize=(10, 5))
sns.barplot(x=categories, y=precision_values)
plt.ylabel("Precision")
plt.title("Precision theo từng lớp - SVM")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ====== Huấn luyện mô hình Random Forest ======
print("\nĐang huấn luyện mô hình Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ====== Đánh giá trên tập validation ======
y_val_pred_rf = rf_model.predict(X_val)
val_acc_rf = accuracy_score(y_val, y_val_pred_rf)
val_f1_rf = f1_score(y_val, y_val_pred_rf, average='weighted')
print(f"\n[Random Forest] Accuracy (val): {val_acc_rf*100:.2f}%")
print(f"[Random Forest] F1-score (val): {val_f1_rf:.4f}")

# ====== Đánh giá trên tập test ======
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f"\n[Random Forest] Accuracy (test): {acc_rf*100:.2f}%")
print(f"[Random Forest] F1-score (test): {f1_rf:.4f}")
print("\n[Random Forest] Báo cáo phân loại:")
print(classification_report(y_test, y_pred_rf, target_names=categories))

# ====== Vẽ ma trận nhầm lẫn cho Random Forest ======
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=categories)
disp_rf.plot(cmap="Greens", xticks_rotation=45)
plt.title("Ma trận nhầm lẫn - Random Forest")
plt.tight_layout()
plt.show()

# Du doan anh
def predict_image(image_path, model, scaler):
    # Đọc và xử lý ảnh
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Trích xuất HOG
    hog_feature = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)

    # Chuẩn hóa
    hog_feature = scaler.transform([hog_feature])

    # Dự đoán
    prediction = model.predict(hog_feature)
    return prediction[0]

# Load model và scaler
svm_model = joblib.load("/content/drive/MyDrive/Train_Lua_5/svm_model.pkl")
scaler = joblib.load("/content/drive/MyDrive/Train_Lua_5/scaler.pkl")

# Test với một ảnh mới
image_path = "/content/drive/MyDrive/img/img1.png"
label = predict_image(image_path, svm_model, scaler)
print(f"Ảnh này thuộc lớp: {categories[label]}")

