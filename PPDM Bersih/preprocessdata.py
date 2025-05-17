import os
import dlib
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

# Konstanta LBP
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

# Konstanta GLCM (arah)
DIRECTION_OFFSETS = {
    0: (0, 1),
    45: (-1, 1),
    90: (-1, 0),
    135: (-1, -1)
}

dataset_dir = "augmented_dataset"
result_dir = "result"

# Inisialisasi detektor wajah dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def crop(file, image_label):
    image = cv2.imread(file)
    # Jika input adalah path (string), baca gambar
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            return None
    
    # Convert ke grayscale jika belum
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize hanya jika gambar valid
    if gray.size == 0:
        return None
        

    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)
    faces = detector(gray_equalized, 0)

    if len(faces) == 0:
        return None

    for face in faces:
        landmarks = predictor(gray_equalized, face)

        def get_coords(idxs):
            return [(landmarks.part(i).x, landmarks.part(i).y) for i in idxs]

        def create_mask(coords, shape):
            mask = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(coords, np.int32)], 255)
            return mask
        
        parts = {
        "left_eye" : range(36, 42),
        "right_eye" : range(42, 48),
        "left_eyebrow" : range(17, 22),
        "right_eyebrow" : range(22, 27),
        "mouth" : range(46, 68),
        "jawline" : range(0, 17),
        "nose" : range(27, 36),
        }

        mask = sum([create_mask(get_coords(part), gray.shape) for part in parts.values()])
        extracted = cv2.bitwise_and(gray, gray, mask=mask)
        return extracted

def compute_glcm_manual(image, distance=1, levels=256):
    h, w = image.shape
    glcm_list = []

    for dy, dx in DIRECTION_OFFSETS.values():
        glcm = np.zeros((levels, levels), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                ni, nj = i + dy * distance, j + dx * distance
                if 0 <= ni < h and 0 <= nj < w:
                    glcm[image[i, j], image[ni, nj]] += 1
        glcm = glcm + glcm.T
        glcm = glcm / (glcm.sum() + 1e-6)
        glcm_list.append(glcm)
    return glcm_list

def extract_glcm_features(image):
    features = []
    for glcm in compute_glcm_manual(image):
        contrast = dissimilarity = homogeneity = asm = correlation = 0.0
        mean_i = mean_j = 0.0

        for i in range(256):
            for j in range(256):
                pij = glcm[i, j]
                mean_i += i * pij
                mean_j += j * pij

        std_i = std_j = 0.0
        for i in range(256):
            for j in range(256):
                pij = glcm[i, j]
                std_i += pij * ((i - mean_i) ** 2)
                std_j += pij * ((j - mean_j) ** 2)

        std_i = np.sqrt(std_i)
        std_j = np.sqrt(std_j)

        for i in range(256):
            for j in range(256):
                pij = glcm[i, j]
                contrast += pij * ((i - j) ** 2)
                dissimilarity += pij * abs(i - j)
                homogeneity += pij / (1 + abs(i - j))
                asm += pij ** 2
                if std_i > 0 and std_j > 0:
                    correlation += ((i - mean_i) * (j - mean_j) * pij) / (std_i * std_j)

        energy = np.sqrt(asm)
        features += [dissimilarity, correlation, homogeneity, contrast, asm, energy]
    return features

def extract_hog_features_manual(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    bins = np.floor(angle / (180/9)).astype(np.int32)
    bins[bins >= 9] = 8

    hist = np.array([np.sum(magnitude[bins==i]) for i in range(9)], dtype=np.float32)
    return (hist / (hist.sum() + 1e-6)).tolist()

def extract_lbp_features_manual(image):
    h, w = image.shape
    lbp_image = np.zeros_like(image)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = image[i, j]
            binary = ''.join(['1' if image[i + dx, j + dy] >= center else '0'
                              for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]])
            lbp_image[i, j] = int(binary, 2)

    hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
    return (hist / (hist.sum() + 1e-6)).tolist()

def extract_all_features(image, label):
    return extract_glcm_features(image) + extract_hog_features_manual(image) + extract_lbp_features_manual(image) + [label]

def normalize_features(df):
    features = df.drop(columns=["label"])
    labels = df["label"]

    scaled_features = MinMaxScaler().fit_transform(features)
    df_normalized = pd.DataFrame(scaled_features, columns=features.columns)
    df_normalized["label"] = labels.values
    return df_normalized

def one_hot_encode_labels(labels_column):
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(labels_column)
    return encoded, encoder

def split_and_save(X, y, result_dir="result", test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y.argmax(axis=1))  # ← Sudah di-indent
    
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, "X_train.npy"), X_train)
    np.save(os.path.join(result_dir, "X_test.npy"), X_test)
    np.save(os.path.join(result_dir, "y_train.npy"), y_train)
    np.save(os.path.join(result_dir, "y_test.npy"), y_test)
    
    print("✅ Dataset berhasil dibagi dan disimpan ke folder.")


if __name__ == "__main__":
    print("\n⏳ Memulai proses segmentasi wajah dan ekstraksi fitur...")
    print("=======================================================")
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n🔍 Memproses gambar dari folder: {dataset_dir}")
    features_all = []
    for folder in os.listdir(dataset_dir):
        print(f"📂 Mengekstraksi fitur dari kelas: {folder}")
        folder_path = os.path.join(dataset_dir, folder)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            cropped_img = crop(img_path, folder)
            if cropped_img is not None:
                features_all.append(extract_all_features(cropped_img, folder)) 

    print("\n✅ Ekstraksi fitur selesai!")
    print(f"📊 Total sampel yang diproses: {len(features_all)}")

    # Simpan fitur mentah & normalisasi
    glcm_cols = [f"{prop}_{angle}" for prop in ["dissimilarity", "correlation", "homogeneity", "contrast", "ASM", "energy"] for angle in ["0", "45", "90", "135"]]
    hog_cols = [f"hog_{i}" for i in range(9)]
    lbp_cols = [f"lbp_{i}" for i in range(256)]
    columns = glcm_cols + hog_cols + lbp_cols + ["label"]

    df = pd.DataFrame(features_all, columns=columns)
    df.to_csv(os.path.join(result_dir, "combined_features.csv"), index=False)
    print("\n⚖️ Melakukan normalisasi fitur...")
    df_normalized = normalize_features(df)
    print("✅ Normalisasi selesai!")
    df_normalized.to_csv(os.path.join(result_dir, "normalized_features.csv"), index=False)
    print("🧪 Ekstraksi & normalisasi fitur selesai.")
    
    print("\n🔢 Melakukan one-hot encoding label...")
    # One-hot encode label & split data
    y_encoded, encoder = one_hot_encode_labels(df_normalized[["label"]])
    joblib.dump(encoder, os.path.join(result_dir, "onehot_encoder.pkl"))

    X = df_normalized.drop(columns=["label"]).values
    split_and_save(X, y_encoded, result_dir)

    print("✅ Preprocessing lengkap selesai!")
