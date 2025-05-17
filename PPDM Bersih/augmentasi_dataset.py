import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance

# Path
input_dir = "dataset"
output_dir = "augmented_dataset"

# Jumlah target augmentasi per kelas (atur agar seimbang, misal 1000 per kelas)
target_count = 4000

# Fungsi augmentasi
def augment_image(img):
    aug_images = []

    # 1. Flip horizontal
    aug_images.append(cv2.flip(img, 1))

    # 2. Rotate +15 dan -15
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [15, -15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        aug_images.append(rotated)

    # 3. Brightness adjustment
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for factor in [0.8, 1.2]:
        enhancer = ImageEnhance.Brightness(pil_img)
        bright = enhancer.enhance(factor)
        aug_images.append(cv2.cvtColor(np.array(bright), cv2.COLOR_RGB2BGR))

    return aug_images

# Proses augmentasi per kelas
for class_name in os.listdir(input_dir):
    input_class_path = os.path.join(input_dir, class_name)
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    images = os.listdir(input_class_path)
    existing_count = len(images)

    # Salin gambar asli dulu ke output
    for img_name in images:
        src_path = os.path.join(input_class_path, img_name)
        dst_path = os.path.join(output_class_path, img_name)
        img = cv2.imread(src_path)
        cv2.imwrite(dst_path, img)

    # Lanjutkan augmentasi jika jumlah gambar masih kurang dari target
    idx = 0
    print(f" Memproses kelas '{class_name}' (sudah {existing_count}, target {target_count})")
    while len(os.listdir(output_class_path)) < target_count:
        img_name = images[idx % existing_count]
        src_path = os.path.join(input_class_path, img_name)
        img = cv2.imread(src_path)

        augmented_imgs = augment_image(img)

        for aug in augmented_imgs:
            new_filename = f"aug_{idx}_{np.random.randint(10000)}.jpg"
            save_path = os.path.join(output_class_path, new_filename)
            cv2.imwrite(save_path, aug)

            if len(os.listdir(output_class_path)) >= target_count:
                break
        idx += 1

print(" Augmentasi selesai. Dataset seimbang tersimpan di:", output_dir)
