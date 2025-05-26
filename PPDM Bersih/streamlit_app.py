import streamlit as st
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import io

# Import fungsi dari file yang sudah ada
from preprocessdata import (
    crop, extract_glcm_features, extract_hog_features_manual, 
    extract_lbp_features_manual, normalize_features
)
from ann_model import NeuralNetwork

# Konfigurasi halaman
st.set_page_config(
    page_title="Emotion Classification",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache untuk loading model dan scaler
@st.cache_resource
def load_model_and_encoder():
    """Load model dan encoder yang sudah dilatih"""
    try:
        # Load model ANN
        model = NeuralNetwork(289, [128, 64], 7, 'relu')  # Sesuaikan dengan arsitektur terbaik
        model.load_model('models/best_model.npy')
        
        # Load encoder
        encoder = joblib.load('result/onehot_encoder.pkl')
        
        # Load data asli (sebelum normalisasi) untuk mendapatkan min-max yang benar
        df_original = pd.read_csv('result/combined_features.csv')
        df_normalized = pd.read_csv('result/normalized_features.csv')
        
        return model, encoder, df_original, df_normalized
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def extract_features_from_image(image):
    """Ekstraksi fitur dari gambar yang diupload"""
    try:
        # Konversi PIL Image ke array numpy
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Crop wajah menggunakan fungsi yang sudah ada
        cropped_face = crop_face_from_array(image_array)
        
        if cropped_face is None:
            return None, "Wajah tidak terdeteksi dalam gambar"
        
        # Ekstraksi fitur
        glcm_features = extract_glcm_features(cropped_face)
        hog_features = extract_hog_features_manual(cropped_face)
        lbp_features = extract_lbp_features_manual(cropped_face)
        
        # Gabungkan semua fitur
        all_features = glcm_features + hog_features + lbp_features
        
        return np.array(all_features), cropped_face
    
    except Exception as e:
        return None, f"Error dalam ekstraksi fitur: {str(e)}"

def crop_face_from_array(image_array):
    """Modifikasi fungsi crop untuk bekerja dengan array numpy"""
    import dlib
    
    # Inisialisasi detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Convert ke grayscale jika perlu
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()
    
    # Resize
    image_resized = cv2.resize(image_array, (128, 128))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY) if len(image_resized.shape) == 3 else cv2.resize(gray, (128, 128))
    gray_equalized = cv2.equalizeHist(gray)
    
    # Deteksi wajah
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
            "left_eye": range(36, 42),
            "right_eye": range(42, 48),
            "left_eyebrow": range(17, 22),
            "right_eyebrow": range(22, 27),
            "mouth": range(46, 68),
            "jawline": range(0, 17),
            "nose": range(27, 36),
        }
        
        mask = sum([create_mask(get_coords(part), gray.shape) for part in parts.values()])
        extracted = cv2.bitwise_and(gray, gray, mask=mask)
        return extracted

def normalize_single_sample(features, original_df):
    """Normalisasi fitur untuk satu sampel berdasarkan data training asli"""
    # Buat kolom names yang sama dengan data training
    glcm_cols = [f"{prop}_{angle}" for prop in ["dissimilarity", "correlation", "homogeneity", "contrast", "ASM", "energy"] for angle in ["0", "45", "90", "135"]]
    hog_cols = [f"hog_{i}" for i in range(9)]
    lbp_cols = [f"lbp_{i}" for i in range(256)]
    feature_columns = glcm_cols + hog_cols + lbp_cols
    
    # Pastikan jumlah fitur sesuai
    if len(features) != len(feature_columns):
        st.error(f"Jumlah fitur tidak sesuai: {len(features)} vs {len(feature_columns)}")
        return None
    
    # Buat DataFrame dari fitur
    feature_df = pd.DataFrame([features], columns=feature_columns)
    
    # Normalisasi menggunakan min-max dari data training ASLI (bukan yang sudah dinormalisasi)
    normalized_features = []
    training_features = original_df.drop(columns=['label'])
    
    for i, col in enumerate(feature_columns):
        if col in training_features.columns:
            min_val = training_features[col].min()
            max_val = training_features[col].max()
            
            # Hindari pembagian dengan nol
            if max_val == min_val:
                normalized_val = 0.0
            else:
                normalized_val = (features[i] - min_val) / (max_val - min_val)
                # Clip ke range [0, 1]
                normalized_val = np.clip(normalized_val, 0, 1)
            
            normalized_features.append(normalized_val)
        else:
            st.error(f"Kolom {col} tidak ditemukan dalam data training")
            return None
    
    return np.array(normalized_features)

def plot_prediction_confidence(probabilities, class_names):
    """Plot confidence score untuk setiap kelas emosi"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Buat bar plot
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(class_names, probabilities, color=colors)
    
    # Highlight prediksi tertinggi
    max_idx = np.argmax(probabilities)
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(0.8)
    
    ax.set_ylabel('Confidence Score')
    ax.set_title('Emotion Classification Confidence')
    ax.set_ylim(0, 1)
    
    # Tambahkan value labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def main():
    st.title("🎭 Emotion Classification from Face Images")
    st.markdown("Upload gambar wajah untuk mendeteksi emosi menggunakan Neural Network")
    
    # Load model dan encoder
    model, encoder, original_df, normalized_df = load_model_and_encoder()
    
    if model is None:
        st.error("Gagal memuat model. Pastikan file model dan encoder tersedia.")
        return
    
    # Sidebar untuk informasi
    with st.sidebar:
        st.header("ℹ️ Informasi")
        st.write("**Kelas Emosi yang Dapat Dideteksi:**")
        if encoder is not None:
            for i, emotion in enumerate(encoder.categories_[0]):
                st.write(f"• {emotion}")
        
        st.write("**Format Gambar yang Didukung:**")
        st.write("• JPG, JPEG, PNG")
        st.write("• Gambar harus mengandung wajah yang jelas")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Gambar")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar wajah untuk klasifikasi emosi"
        )
        
        if uploaded_file is not None:
            # Tampilkan gambar asli
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diupload", use_container_width=True)
            
            # Tombol untuk mulai klasifikasi
            if st.button("🔍 Klasifikasi Emosi", type="primary"):
                with st.spinner("Memproses gambar..."):
                    # Ekstraksi fitur
                    features, cropped_face = extract_features_from_image(image)
                    
                    if features is None:
                        st.error(f"❌ {cropped_face}")
                        return
                    
                    # Normalisasi fitur
                    try:
                        normalized_features = normalize_single_sample(features, original_df)
                        
                        if normalized_features is None:
                            st.error("❌ Error dalam normalisasi fitur")
                            return
                        
                        # Debug: Tampilkan beberapa nilai fitur
                        with st.expander("🔍 Debug Info"):
                            st.write(f"Jumlah fitur: {len(features)}")
                            st.write(f"Fitur mentah (5 pertama): {features[:5]}")
                            st.write(f"Fitur ternormalisasi (5 pertama): {normalized_features[:5]}")
                            st.write(f"Min-Max fitur ternormalisasi: {normalized_features.min():.4f} - {normalized_features.max():.4f}")
                        
                        # Prediksi dengan debugging
                        prediction_probs = model.forward(normalized_features.reshape(1, -1))[0]
                        predicted_class_idx = np.argmax(prediction_probs)
                        predicted_emotion = encoder.categories_[0][predicted_class_idx]
                        confidence = prediction_probs[predicted_class_idx]
                        
                        # Debug: Tampilkan probabilitas
                        with st.expander("🎯 Debug Prediksi"):
                            st.write("Probabilitas untuk setiap kelas:")
                            for i, (emotion, prob) in enumerate(zip(encoder.categories_[0], prediction_probs)):
                                st.write(f"  {emotion}: {prob:.4f}")
                            st.write(f"Kelas terprediksi: {predicted_class_idx} ({predicted_emotion})")
                        
                        # Simpan hasil ke session state
                        st.session_state.prediction_results = {
                            'emotion': predicted_emotion,
                            'confidence': confidence,
                            'probabilities': prediction_probs,
                            'cropped_face': cropped_face,
                            'class_names': encoder.categories_[0],
                            'raw_features': features,
                            'normalized_features': normalized_features
                        }
                        
                        st.success("✅ Klasifikasi berhasil!")
                        
                    except Exception as e:
                        st.error(f"❌ Error dalam prediksi: {str(e)}")
                        st.write("Detail error:", e)
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        st.header("📊 Hasil Klasifikasi")
        
        # Tampilkan hasil jika ada
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            
            # Tampilkan wajah yang sudah di-crop
            if results['cropped_face'] is not None:
                st.subheader("🎯 Wajah yang Dideteksi")
                # Convert grayscale to RGB untuk display yang lebih baik
                cropped_display = cv2.cvtColor(results['cropped_face'], cv2.COLOR_GRAY2RGB) if len(results['cropped_face'].shape) == 2 else results['cropped_face']
                st.image(cropped_display, caption="Segmented Face", use_container_width=True)
            
            # Hasil prediksi utama
            st.subheader("🎭 Prediksi Emosi")
            col_emotion, col_conf = st.columns(2)
            
            with col_emotion:
                st.metric("Emosi Terdeteksi", results['emotion'])
            
            with col_conf:
                st.metric("Confidence", f"{results['confidence']:.1%}")
            
            # Progress bar untuk confidence
            st.progress(results['confidence'])
            
            # Grafik confidence untuk semua kelas
            st.subheader("📈 Confidence Score untuk Semua Emosi")
            fig = plot_prediction_confidence(results['probabilities'], results['class_names'])
            st.pyplot(fig)
            
            # Tabel detail probabilitas
            with st.expander("📋 Detail Probabilitas"):
                prob_df = pd.DataFrame({
                    'Emosi': results['class_names'],
                    'Probabilitas': results['probabilities'],
                    'Persentase': [f"{p:.1%}" for p in results['probabilities']]
                })
                prob_df = prob_df.sort_values('Probabilitas', ascending=False)
                st.dataframe(prob_df, use_container_width=True)
        
        else:
            st.info("👆 Upload gambar dan klik tombol klasifikasi untuk melihat hasil")

if __name__ == "__main__":
    main()
