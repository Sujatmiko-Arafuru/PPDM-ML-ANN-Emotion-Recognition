import streamlit as st
import os
import numpy as np
import cv2
import dlib
import joblib
import pandas as pd
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Import the NeuralNetwork class from ann_model.py
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu'):
        """
        Inisialisasi Arsitektur Neural Network
        
        Parameters:
        -----------
        input_size : int
            Jumlah neuron pada input layer (jumlah fitur)
        hidden_layers : list
            List berisi jumlah neuron untuk setiap hidden layer
        output_size : int
            Jumlah neuron pada output layer (jumlah kelas)
        activation : str
            Fungsi aktivasi yang digunakan ('relu' atau 'tanh')
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        
        # Layer sizes (termasuk input dan output)
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Inisialisasi weights dan biases
        self.weights = []
        self.biases = []
    
    def relu(self, x):
        """Fungsi aktivasi ReLU"""
        return np.maximum(0, x)
    
    def tanh(self, x):
        """Fungsi aktivasi tanh"""
        return np.tanh(x)
    
    def softmax(self, x):
        """Fungsi aktivasi softmax untuk output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data dengan shape (n_samples, input_size)
            
        Returns:
        --------
        numpy.ndarray
            Output dari network, probabilitas untuk setiap kelas
        """
        self.z_values = []
        self.a_values = [X]  # Input layer activation
        
        # Propagasi melalui hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Aktifasi dengan fungsi yang dipilih
            if self.activation == 'relu':
                a = self.relu(z)
            else:  # 'tanh'
                a = self.tanh(z)
                
            self.a_values.append(a)
        
        # Output layer dengan softmax
        z_out = np.dot(self.a_values[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        
        # Softmax activation for output layer
        output = self.softmax(z_out)
        self.a_values.append(output)
        
        return output
    
    def predict(self, X):
        """
        Melakukan prediksi
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Predicted class indices
        """
        output = self.forward(X)
        return np.argmax(output, axis=1), output
    
    def load_model(self, filepath):
        """Load model parameters from file"""
        try:
            model_params = np.load(filepath, allow_pickle=True).item()
            self.weights = model_params['weights']
            self.biases = model_params['biases']
            self.layer_sizes = model_params['layer_sizes']
            self.activation = model_params['activation']
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

# Constants from preprocessdata.py
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

DIRECTION_OFFSETS = {
    0: (0, 1),
    45: (-1, 1),
    90: (-1, 0),
    135: (-1, -1)
}

# Initialize face detection tools
@st.cache_resource
def load_face_detector():
    try:
        detector = dlib.get_frontal_face_detector()
        # Check if the shape predictor file exists, otherwise download it
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        if not os.path.exists(predictor_path):
            st.warning("Face landmark predictor model not found. Please upload the shape_predictor_68_face_landmarks.dat file.")
            return None, None
        
        predictor = dlib.shape_predictor(predictor_path)
        return detector, predictor
    except Exception as e:
        st.error(f"Failed to load face detector: {e}")
        return None, None

def preprocess_for_expression(image):
    """Preprocessing to help with expressive faces"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return filtered

def crop(image, detector, predictor):
    """More robust face cropping function"""
    if detector is None or predictor is None:
        return None
        
    # Preprocess image for better detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)
    
    # Try detecting with different scales
    for scale in [1.0, 1.25, 0.8]:
        faces = detector(gray_equalized, int(scale))
        if len(faces) > 0:
            break
    
    if len(faces) == 0:
        return None
    
    # Get the largest face
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    # Get landmarks
    landmarks = predictor(gray_equalized, face)
    
    # Get face coordinates
    x_coords = [landmarks.part(i).x for i in range(68)]
    y_coords = [landmarks.part(i).y for i in range(68)]
    
    # Expand the bounding box slightly
    x_min, x_max = max(0, min(x_coords)-15), min(image.shape[1], max(x_coords)+15)
    y_min, y_max = max(0, min(y_coords)-15), min(image.shape[0], max(y_coords)+15)
    
    # Crop and resize the face
    cropped = gray_equalized[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return None
    
    return cv2.resize(cropped, (128, 128))

def compute_glcm_manual(image, distance=1, levels=256):
    """Compute Gray Level Co-occurrence Matrix manually"""
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
    """Extract GLCM features from image"""
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
    """Extract HOG features from image"""
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    bins = np.floor(angle / (180/9)).astype(np.int32)
    bins[bins >= 9] = 8
    
    hist = np.array([np.sum(magnitude[bins==i]) for i in range(9)], dtype=np.float32)
    return (hist / (hist.sum() + 1e-6)).tolist()

def extract_lbp_features_manual(image):
    """Extract LBP features from image"""
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

def extract_all_features(image):
    """Extract all features from image"""
    try:
        glcm_features = extract_glcm_features(image)
        hog_features = extract_hog_features_manual(image)
        lbp_features = extract_lbp_features_manual(image)
        return glcm_features + hog_features + lbp_features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def get_feature_names():
    """Get feature names for visualization"""
    glcm_cols = [f"{prop}_{angle}" for prop in ["dissimilarity", "correlation", "homogeneity", "contrast", "ASM", "energy"] for angle in ["0", "45", "90", "135"]]
    hog_cols = [f"hog_{i}" for i in range(9)]
    lbp_cols = [f"lbp_{i}" for i in range(256)]
    return glcm_cols + hog_cols + lbp_cols

@st.cache_resource
def load_model_and_encoder():
    """Load the model and encoder"""
    try:
        # Create model directory if not exists
        os.makedirs("models", exist_ok=True)
        
        # Initialize model with some default structure (will be overwritten when loading)
        model = NeuralNetwork(289, [128, 64], 7, 'relu')  # Placeholder values
        
        # Check if model file exists, otherwise show warning
        model_path = "models/best_model.npy"
        if not os.path.exists(model_path):
            st.warning("Model file not found. Please upload the best_model.npy file.")
            return None, None
            
        # Load model
        model_loaded = model.load_model(model_path)
        if not model_loaded:
            return None, None
            
        # Check if encoder file exists, otherwise show warning
        encoder_path = "result/onehot_encoder.pkl"
        if not os.path.exists(encoder_path):
            st.warning("Encoder file not found. Please upload the onehot_encoder.pkl file.")
            return model, None
            
        # Load encoder
        encoder = joblib.load(encoder_path)
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model or encoder: {e}")
        return None, None

def load_scaler():
    """Load MinMaxScaler for feature normalization"""
    try:
        # Check if the scaler exists, otherwise create one
        scaler_path = "result/feature_scaler.pkl"
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
        else:
            st.warning("Feature scaler not found. Using default normalization.")
            return None
    except Exception as e:
        st.error(f"Error loading feature scaler: {e}")
        return None

def plot_feature_importance(features, feature_names, top_n=20):
    """Plot top N most important features"""
    # Get absolute values of features to estimate importance
    feature_importance = np.abs(features)
    
    # Get indices of top N features
    top_indices = np.argsort(feature_importance)[-top_n:]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Feature': [feature_names[i] for i in top_indices],
        'Value': [features[i] for i in top_indices]
    })
    
    # Sort by absolute value
    df = df.sort_values('Value', key=abs, ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red' if x < 0 else 'green' for x in df['Value']]
    ax = sns.barplot(data=df, x='Value', y='Feature', palette=colors)
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    
    return fig

def plot_prediction_probabilities(probabilities, class_names):
    """Plot prediction probabilities for each class"""
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': probabilities[0]
    })
    
    df = df.sort_values('Probability', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Probability', y='Class')
    plt.title('Prediction Probabilities by Class')
    plt.xlim(0, 1)
    
    for index, row in df.iterrows():
        plt.text(row.Probability + 0.01, index, f'{row.Probability:.4f}')
    
    plt.tight_layout()
    return fig

def create_results_image(original_img, cropped_face, prediction, class_names):
    """Create a visual representation of results"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    ax[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Display cropped face
    if cropped_face is not None:
        ax[1].imshow(cropped_face, cmap='gray')
        ax[1].set_title(f"Detected Face\nPrediction: {class_names[prediction]}")
    else:
        ax[1].text(0.5, 0.5, "No face detected", horizontalalignment='center')
        ax[1].set_title("Face Detection Failed")
    ax[1].axis('off')
    
    plt.tight_layout()
    return fig

# Streamlit app
def main():
    st.set_page_config(
        page_title="Enhanced ANN Face Recognition Model", 
        page_icon="🧠", 
        layout="wide"
    )
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This enhanced app uses an Artificial Neural Network (ANN) model to recognize faces, "
        "with improved detection for expressive faces like anger."
    )
    
    st.sidebar.title("Model Configuration")
    # Create file uploaders for model and encoder files
    model_file = st.sidebar.file_uploader("Upload model file (best_model.npy)", type=["npy"])
    encoder_file = st.sidebar.file_uploader("Upload encoder file (onehot_encoder.pkl)", type=["pkl"])
    predictor_file = st.sidebar.file_uploader("Upload shape predictor file (shape_predictor_68_face_landmarks.dat)", type=["dat"])
    
    # Save uploaded files
    if model_file:
        os.makedirs("models", exist_ok=True)
        with open("models/best_model.npy", "wb") as f:
            f.write(model_file.getbuffer())
        st.sidebar.success("Model file uploaded successfully!")
        
    if encoder_file:
        os.makedirs("result", exist_ok=True)
        with open("result/onehot_encoder.pkl", "wb") as f:
            f.write(encoder_file.getbuffer())
        st.sidebar.success("Encoder file uploaded successfully!")
        
    if predictor_file:
        with open("shape_predictor_68_face_landmarks.dat", "wb") as f:
            f.write(predictor_file.getbuffer())
        st.sidebar.success("Shape predictor file uploaded successfully!")
    
    # Main content
    st.title("Enhanced Face Recognition with ANN")
    st.markdown("""
    This enhanced application uses an Artificial Neural Network to recognize faces, with improved handling of expressive faces.
    
    **Improvements made:**
    - Better face detection for expressive faces (like anger)
    - Enhanced preprocessing for difficult images
    - More robust cropping algorithm
    
    **How to use:**
    1. Upload an image file containing a face
    2. The app will detect the face, extract features, and make a prediction
    3. View the results and the feature analysis
    """)
    
    # Load detector, model, and encoder
    detector, predictor = load_face_detector()
    model, encoder = load_model_and_encoder()
    scaler = load_scaler()
    
    # Check if all required components are loaded
    all_components_loaded = True
    
    if detector is None or predictor is None:
        st.warning("⚠️ Face detector not loaded. Please upload the shape predictor file.")
        all_components_loaded = False
        
    if model is None:
        st.warning("⚠️ Model not loaded. Please upload the model file.")
        all_components_loaded = False
        
    if encoder is None:
        st.warning("⚠️ Encoder not loaded. Please upload the encoder file.")
        all_components_loaded = False
    
    # Get class names if encoder is available
    class_names = []
    if encoder is not None:
        try:
            class_names = encoder.categories_[0].tolist()
            st.sidebar.markdown("## Classes")
            for i, name in enumerate(class_names):
                st.sidebar.write(f"{i}: {name}")
        except:
            st.sidebar.warning("Couldn't extract class names from encoder")
    
    # File uploader for test image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Show original image
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        
        # Check if all components are loaded before processing
        if all_components_loaded:
            with st.spinner("Processing image with enhanced detection..."):
                # Process image
                try:
                    # Preprocess for expression
                    preprocessed = preprocess_for_expression(image)
                    
                    # Convert back to 3-channel for display (but use grayscale for processing)
                    display_img = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
                    
                    # Crop face with the new function
                    cropped_face = crop(display_img, detector, predictor)
                    
                    if cropped_face is None:
                        # Try alternative approach with just face detection (no landmarks)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        faces = detector(gray, 1)
                        if len(faces) > 0:
                            face = faces[0]
                            cropped_face = gray[face.top():face.bottom(), face.left():face.right()]
                            cropped_face = cv2.resize(cropped_face, (128, 128))
                            st.warning("Used simple face detection (landmarks might be inaccurate)")
                        else:
                            st.error("No face detected in the image. Please try another image.")
                    else:
                        # Show cropped face
                        st.image(cropped_face, caption="Detected Face", width=300)
                        
                        # Extract features
                        features = extract_all_features(cropped_face)
                        
                        if features is None:
                            st.error("Failed to extract features from the image.")
                        else:
                            # Create DataFrame for features
                            feature_names = get_feature_names()
                            features_df = pd.DataFrame([features], columns=feature_names)
                            
                            # Normalize features if scaler is available
                            if scaler is not None:
                                features_normalized = scaler.transform(features_df)
                            else:
                                # Simple min-max scaling as fallback
                                features_min = features_df.min()
                                features_max = features_df.max()
                                features_normalized = (features_df - features_min) / (features_max - features_min + 1e-10)
                            
                            # Make prediction
                            prediction_idx, prediction_probs = model.predict(features_normalized.values)
                            prediction_idx = prediction_idx[0]
                            
                            # Show prediction
                            st.success(f"Prediction: {class_names[prediction_idx]}")
                            
                            # Create tabs for different visualizations
                            tab1, tab2, tab3 = st.tabs(["Prediction Results", "Feature Importance", "Raw Features"])
                            
                            with tab1:
                                # Plot prediction probabilities
                                prob_fig = plot_prediction_probabilities(prediction_probs, class_names)
                                st.pyplot(prob_fig)
                                
                                # Show results image
                                results_fig = create_results_image(image, cropped_face, prediction_idx, class_names)
                                st.pyplot(results_fig)
                            
                            with tab2:
                                # Plot feature importance
                                importance_fig = plot_feature_importance(features, feature_names, top_n=20)
                                st.pyplot(importance_fig)
                            
                            with tab3:
                                # Show raw features
                                st.write("Raw Feature Values")
                                st.dataframe(features_df)
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
    
    # Instructions section
    st.markdown("---")
    st.header("Enhanced Detection Features")
    st.markdown("""
    ### Key Improvements
    1. **Better Expression Handling**: Improved detection for faces showing strong expressions like anger
    2. **Advanced Preprocessing**: CLAHE and bilateral filtering for better feature extraction
    3. **Robust Cropping**: Multiple detection attempts with different scales and expanded bounding boxes
    
    ### Required Files
    To use this application, you need to upload these files:
    1. **best_model.npy** - The trained ANN model file
    2. **onehot_encoder.pkl** - The one-hot encoder for class labels
    3. **shape_predictor_68_face_landmarks.dat** - Dlib's face landmark predictor
    
    ### Troubleshooting
    - If detection fails, try images with clearer facial features
    - For anger expressions, ensure the face is reasonably frontal
    - Good lighting and minimal obstructions improve results
    """)
    
    # Download example button
    if st.button("Download Example Files"):
        st.markdown("""
        You can download the required files from these sources:
        - **shape_predictor_68_face_landmarks.dat**: Download from the official dlib website
        - **best_model.npy** and **onehot_encoder.pkl**: These should be generated from your training process
        """)

if __name__ == "__main__":
    main()
