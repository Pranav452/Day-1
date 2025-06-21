import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Individual Implementation for Perceptron From Scratch Assignment
# Author: Pranav
# Custom fruit classification with unique DJ-knob learning rate analogy

class PerceptronFromScratch:
    def __init__(self, learning_rate=0.01, max_epochs=500, target_loss=0.05):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.target_loss = target_loss
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.accuracy_history = []
        
    def sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def binary_cross_entropy(self, y_true, y_pred):
        """Binary cross-entropy loss function"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def forward_pass(self, X):
        """Forward pass through the perceptron"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_gradients(self, X, y_true, y_pred):
        """Compute gradients for weights and bias"""
        m = X.shape[0]  # number of samples
        
        # Gradient of loss with respect to weights
        dw = (1/m) * np.dot(X.T, (y_pred - y_true))
        
        # Gradient of loss with respect to bias
        db = (1/m) * np.sum(y_pred - y_true)
        
        return dw, db
    
    def calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy by converting probabilities to predictions"""
        predictions = (y_pred >= 0.5).astype(int)
        return np.mean(predictions == y_true)
    
    def fit(self, X, y):
        """Train the perceptron using batch gradient descent"""
        # Initialize weights and bias
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        print(f"Starting training with learning rate: {self.learning_rate}")
        print(f"Initial random weights: {self.weights}")
        print(f"Initial bias: {self.bias}")
        
        # Make initial prediction to see how bad it is
        initial_pred = self.forward_pass(X)
        initial_loss = self.binary_cross_entropy(y, initial_pred)
        initial_accuracy = self.calculate_accuracy(y, initial_pred)
        
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Initial accuracy: {initial_accuracy:.4f}")
        print("-" * 50)
        
        # Training loop
        for epoch in range(self.max_epochs):
            # Forward pass
            y_pred = self.forward_pass(X)
            
            # Calculate loss
            loss = self.binary_cross_entropy(y, y_pred)
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(y, y_pred)
            
            # Store metrics
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1:3d}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
            
            # Early stopping if target loss is reached
            if loss < self.target_loss:
                print(f"Target loss {self.target_loss} reached at epoch {epoch + 1}")
                break
        
        print("-" * 50)
        print(f"Final loss: {loss:.4f}")
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Final weights: {self.weights}")
        print(f"Final bias: {self.bias}")
        
    def predict(self, X):
        """Make predictions on new data"""
        probabilities = self.forward_pass(X)
        predictions = (probabilities >= 0.5).astype(int)
        return predictions, probabilities
    
    def plot_training_history(self):
        """Plot loss and accuracy over epochs"""
        epochs = range(1, len(self.loss_history) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(epochs, self.loss_history, 'b-', linewidth=2, label='Training Loss')
        ax1.set_title('Loss Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Binary Cross-Entropy Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(epochs, self.accuracy_history, 'r-', linewidth=2, label='Training Accuracy')
        ax2.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig('training_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

def load_and_preprocess_data(filepath):
    """Load data from CSV and preprocess it"""
    # Load data
    data = pd.read_csv(filepath)
    
    print("Dataset loaded successfully!")
    print(f"Shape: {data.shape}")
    print("\nFirst few rows:")
    print(data.head())
    print(f"\nClass distribution:")
    print(data['label'].value_counts())
    
    # Separate features and labels
    X = data[['length_cm', 'weight_g', 'yellow_score']].values
    y = data['label'].values
    
    # Normalize features to help with convergence
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std
    
    print(f"\nFeature means: {X_mean}")
    print(f"Feature stds: {X_std}")
    
    return X_normalized, y, X_mean, X_std

def main():
    """Main function to run the perceptron experiment"""
    print("=" * 60)
    print("PERCEPTRON FROM SCRATCH - FRUIT CLASSIFICATION")
    print("=" * 60)
    
    # Load and preprocess data
    X, y, X_mean, X_std = load_and_preprocess_data('fruit.csv')
    
    print("\n" + "=" * 60)
    print("TRAINING PERCEPTRON")
    print("=" * 60)
    
    # Create and train perceptron
    perceptron = PerceptronFromScratch(learning_rate=0.1, max_epochs=500, target_loss=0.05)
    perceptron.fit(X, y)
    
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)
    
    # Make predictions on the training data
    predictions, probabilities = perceptron.predict(X)
    
    print("Sample predictions:")
    for i in range(min(10, len(X))):
        actual_label = "Banana" if y[i] == 1 else "Apple"
        pred_label = "Banana" if predictions[i] == 1 else "Apple"
        confidence = probabilities[i] if predictions[i] == 1 else 1 - probabilities[i]
        print(f"Sample {i+1}: Actual={actual_label}, Predicted={pred_label}, Confidence={confidence:.3f}")
    
    print("\n" + "=" * 60)
    print("PLOTTING RESULTS")
    print("=" * 60)
    
    # Plot training history
    perceptron.plot_training_history()
    
    print("Training plots saved as 'training_plots.png'")
    
    # Experiment with different learning rates
    print("\n" + "=" * 60)
    print("LEARNING RATE COMPARISON")
    print("=" * 60)
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    plt.figure(figsize=(12, 8))
    
    for i, lr in enumerate(learning_rates):
        perceptron_lr = PerceptronFromScratch(learning_rate=lr, max_epochs=200)
        perceptron_lr.fit(X, y)
        
        plt.subplot(2, 2, i+1)
        plt.plot(perceptron_lr.loss_history, label=f'LR = {lr}')
        plt.title(f'Loss with Learning Rate = {lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Learning rate comparison saved as 'learning_rate_comparison.png'")

if __name__ == "__main__":
    main() 