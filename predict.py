import joblib
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained models
rf_model = joblib.load("random_forest_model.pkl")  # Load Random Forest model
cnn_model = load_model("cnn_qr_model.h5")  # Load CNN model

def predict_qr(image_path):
    """
    Predict if a QR code is 'First Print' (Original) or 'Second Print' (Counterfeit).
    
    Parameters:
        image_path (str): Path to the input QR code image.

    Returns:
        None: Prints the predictions.
    """
    IMG_SIZE = (256, 256)  # Ensure consistent image size

    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    img_resized = cv2.resize(img, IMG_SIZE)  # Resize to match training data
    img_normalized = img_resized / 255.0  # Normalize pixel values (0-1)

    # Reshape for CNN
    img_cnn = img_normalized.reshape(1, 256, 256, 1)  # Shape: (1, 256, 256, 1)

    # Reshape for Random Forest
    num_features = rf_model.n_features_in_  # Get expected feature size
    img_rf = img_normalized.flatten().reshape(1, -1)  # Shape: (1, num_features)

    # If features are missing, pad with zeros
    if img_rf.shape[1] < num_features:
        padding = np.zeros((1, num_features - img_rf.shape[1]))
        img_rf = np.hstack((img_rf, padding))

    # Make predictions
    rf_pred = rf_model.predict(img_rf)[0]  # Random Forest Prediction
    cnn_pred_prob = cnn_model.predict(img_cnn)[0][0]  # CNN Probability
    cnn_pred = 1 if cnn_pred_prob > 0.5 else 0  # Convert probability to class

    # Map Predictions to Labels
    label_map = {0: "First Print (Original)", 1: "Second Print (Counterfeit)"}
    print(f"Random Forest Prediction: {label_map[rf_pred]}")
    print(f"CNN Prediction: {label_map[cnn_pred]} (Confidence: {cnn_pred_prob:.4f})")

    # Display Image
    plt.imshow(img_resized, cmap='gray')
    plt.title(f"Prediction: {label_map[cnn_pred]}")
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "QR Data/Second Print/input_image_active (4).png"  # Replace with actual image path
    predict_qr(image_path)
