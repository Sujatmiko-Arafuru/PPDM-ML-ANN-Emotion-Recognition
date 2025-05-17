import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Pastikan folder untuk menyimpan model ada
os.makedirs("models", exist_ok=True)

# 1. KARAKTERISTIK NEURAL NETWORK
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
        
        # He initialization untuk ReLU, Xavier/Glorot untuk tanh
        for i in range(len(self.layer_sizes) - 1):
            if activation == 'relu' and i < len(self.layer_sizes) - 2:  # ReLU untuk hidden layers
                scale = np.sqrt(2 / self.layer_sizes[i])  # He initialization
            else:  # tanh atau layer output
                scale = np.sqrt(1 / self.layer_sizes[i])  # Xavier initialization
                
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * scale
            b = np.zeros((1, self.layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Menyimpan nilai aktivasi dan pre-aktivasi untuk backprop
        self.z_values = []  # pre-activation
        self.a_values = []  # activation
        
    # 2. FUNGSI AKTIVASI
    def relu(self, x):
        """Fungsi aktivasi ReLU"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Turunan fungsi aktivasi ReLU"""
        return np.where(x > 0, 1, 0)
    
    def tanh(self, x):
        """Fungsi aktivasi tanh"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Turunan fungsi aktivasi tanh"""
        return 1 - np.tanh(x)**2
    
    def softmax(self, x):
        """Fungsi aktivasi softmax untuk output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # 3. FORWARD PROPAGATION
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
    
    # 4. FUNGSI LOSS
    def categorical_crossentropy(self, y_true, y_pred):
        """
        Categorical Crossentropy Loss
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            One-hot encoded true labels
        y_pred : numpy.ndarray
            Predicted probabilities
            
        Returns:
        --------
        float
            Mean loss value
        """
        # Clip values to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Calculate cross-entropy
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    # 5. BACKWARD PROPAGATION
    def backward(self, X, y):
        """
        Backward propagation untuk menghitung gradien
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        y : numpy.ndarray
            One-hot encoded true labels
            
        Returns:
        --------
        list, list
            Gradient untuk weights dan biases
        """
        m = X.shape[0]  # Jumlah samples
        
        # Inisialisasi dw dan db
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error (derivative of softmax+crossentropy)
        delta = self.a_values[-1] - y  # Simplifikasi dari backprop softmax + crossentropy
        
        # Backprop untuk output layer
        dw[-1] = np.dot(self.a_values[-2].T, delta) / m
        db[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Backprop untuk hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            # Propagate error
            delta = np.dot(delta, self.weights[l+1].T)
            
            # Apply activation derivative
            if self.activation == 'relu':
                delta *= self.relu_derivative(self.z_values[l])
            else:  # 'tanh'
                delta *= self.tanh_derivative(self.z_values[l])
            
            # Calculate gradients
            dw[l] = np.dot(self.a_values[l].T, delta) / m
            db[l] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dw, db
    
    # 6. UPDATE WEIGHTS AND BIASES
    def update_params(self, dw, db, learning_rate):
        """
        Update parameters dengan gradien descent
        
        Parameters:
        -----------
        dw : list
            Gradient untuk weights
        db : list
            Gradient untuk biases
        learning_rate : float
            Learning rate
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dw[i]
            self.biases[i] -= learning_rate * db[i]
    
    # PREDICT FUNCTION
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
        return np.argmax(output, axis=1)
    
    # SAVE AND LOAD MODEL
    def save_model(self, filepath):
        """Save model parameters to file"""
        model_params = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes,
            'activation': self.activation
        }
        np.save(filepath, model_params)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load model parameters from file"""
        model_params = np.load(filepath, allow_pickle=True).item()
        self.weights = model_params['weights']
        self.biases = model_params['biases']
        self.layer_sizes = model_params['layer_sizes']
        self.activation = model_params['activation']
        print(f"Model loaded from {filepath}")

# TRAINING FUNCTION
def train(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
    """
    Fungsi untuk training model
    
    Parameters:
    -----------
    model : NeuralNetwork
        Model neural network
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels (one-hot encoded)
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation labels (one-hot encoded)
    epochs : int
        Jumlah epoch
    batch_size : int
        Ukuran batch
    learning_rate : float
        Learning rate
        
    Returns:
    --------
    dict
        Training history (loss dan accuracy)
    """
    n_samples = X_train.shape[0]
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            output = model.forward(X_batch)
            
            # Backward pass
            dw, db = model.backward(X_batch, y_batch)
            
            # Update parameters
            model.update_params(dw, db, learning_rate)
        
        # Calculate loss and accuracy for training data
        train_output = model.forward(X_train)
        train_loss = model.categorical_crossentropy(y_train, train_output)
        train_pred = np.argmax(train_output, axis=1)
        train_true = np.argmax(y_train, axis=1)
        train_acc = accuracy_score(train_true, train_pred)
        
        # Calculate loss and accuracy for validation data
        val_output = model.forward(X_val)
        val_loss = model.categorical_crossentropy(y_val, val_output)
        val_pred = np.argmax(val_output, axis=1)
        val_true = np.argmax(y_val, axis=1)
        val_acc = accuracy_score(val_true, val_pred)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
    
    history = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies
    }
    
    return history

# EVALUATION FUNCTION
def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluasi model dengan metrics
    
    Parameters:
    -----------
    model : NeuralNetwork
        Model neural network
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels (one-hot encoded)
    class_names : list
        Nama kelas untuk confusion matrix
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification Report
    if class_names is None:
        class_names = [f"Class {i}" for i in range(model.output_size)]
    
    cr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    cr_df = pd.DataFrame(cr).transpose()
    print(cr_df)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Label with counts
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Return metrics
    metrics = {
        'accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': cr
    }
    
    return metrics

# HYPERPARAMETER TUNING
def grid_search(X_train, y_train, X_val, y_val, X_test, y_test, param_grid):
    """
    Grid search untuk hyperparameter tuning
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels (one-hot encoded)
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation labels (one-hot encoded)
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels (one-hot encoded)
    param_grid : dict
        Parameter grid untuk tuning
        
    Returns:
    --------
    dict
        Best parameters and model
    """
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    results = []
    best_val_acc = 0
    best_params = None
    best_model = None
    
    # Permutasi semua kombinasi hyperparameter
    hidden_layers_options = param_grid['hidden_layers']
    activations = param_grid['activation']
    learning_rates = param_grid['learning_rate']
    batch_sizes = param_grid['batch_size']
    epochs_options = param_grid['epochs']
    
    total_combinations = (len(hidden_layers_options) * len(activations) * 
                         len(learning_rates) * len(batch_sizes) * len(epochs_options))
    
    print(f"Total kombinasi yang akan dicoba: {total_combinations}")
    
    combination_count = 0
    start_time = time.time()
    
    # Iterate through all combinations
    for hidden_layers in hidden_layers_options:
        for activation in activations:
            for learning_rate in learning_rates:
                for batch_size in batch_sizes:
                    for epochs in epochs_options:
                        combination_count += 1
                        print(f"\nKombinasi {combination_count}/{total_combinations}:")
                        print(f"Hidden Layers: {hidden_layers}, Activation: {activation}, "
                              f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {epochs}")
                        
                        # Initialize model
                        model = NeuralNetwork(input_size, hidden_layers, output_size, activation)
                        
                        # Train model
                        history = train(model, X_train, y_train, X_val, y_val, 
                                       epochs, batch_size, learning_rate)
                        
                        # Get validation accuracy
                        val_acc = history['val_acc'][-1]
                        
                        # Evaluate on test set
                        test_pred = model.predict(X_test)
                        test_true = np.argmax(y_test, axis=1)
                        test_acc = accuracy_score(test_true, test_pred)
                        
                        # Store results
                        params_dict = {
                            'hidden_layers': hidden_layers,
                            'activation': activation,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'val_accuracy': val_acc,
                            'test_accuracy': test_acc
                        }
                        
                        results.append(params_dict)
                        
                        # Check if this is the best model so far
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_params = params_dict
                            best_model = model
                            
                            # Save best model
                            model.save_model('models/best_model.npy')
                            
                            print(f"New best model found! Validation accuracy: {best_val_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nGrid search completed in {total_time:.2f} seconds.")
    
    # Sort results by validation accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_accuracy', ascending=False)
    results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    
    print("\nTop 5 Parameter Combinations:")
    print(results_df.head(5))
    
    return {
        'best_params': best_params,
        'best_model': best_model,
        'results_df': results_df
    }

# MAIN FUNCTION
def main():
    print("Loading data...")
    # Load data
    X_train = np.load('result/X_train.npy')
    X_test = np.load('result/X_test.npy')
    y_train = np.load('result/y_train.npy')
    y_test = np.load('result/y_test.npy')
    
    print(f"Data loaded - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Load OneHotEncoder to get class names
    encoder = joblib.load('result/onehot_encoder.pkl')
    class_names = encoder.categories_[0]
    
    # Split training data into train and validation sets
    np.random.seed(42)
    validation_split = 0.1  # 10% of training data for validation
    n_train = X_train.shape[0]
    indices = np.random.permutation(n_train)
    n_val = int(n_train * validation_split)
    
    X_val = X_train[indices[:n_val]]
    y_val = y_train[indices[:n_val]]
    X_train_new = X_train[indices[n_val:]]
    y_train_new = y_train[indices[n_val:]]
    
    print(f"Train-validation split - X_train: {X_train_new.shape}, X_val: {X_val.shape}")
    
    # Define parameter grid
    param_grid = {
        'hidden_layers': [
            [64, 32],
            [128, 64],
            [128, 128],
            [256, 128],
            [128, 64, 32],
            [256, 128, 64],
            [512, 256, 128]
        ],
        'activation': ['relu', 'tanh'],
        'learning_rate': [0.01, 0.001, 0.0001],
        'batch_size': [32, 64],
        'epochs': [50, 100, 150]
    }
    
    print("\nStarting hyperparameter tuning...")
    tuning_results = grid_search(X_train_new, y_train_new, X_val, y_val, X_test, y_test, param_grid)
    
    print("\nBest parameters:")
    for key, value in tuning_results['best_params'].items():
        print(f"{key}: {value}")
    
    # Final evaluation of best model
    best_model = tuning_results['best_model']
    print("\nFinal evaluation of best model on test set:")
    metrics = evaluate_model(best_model, X_test, y_test, class_names)
    
    # Plot training history
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
