# QR Code Authentication Classifier

A machine learning and deep learning-based system for detecting counterfeit QR codes. This project uses **Random Forest** and **Convolutional Neural Networks (CNNs)** to classify QR codes as **"First Print" (Original) or "Second Print" (Counterfeit).** The classification pipeline is built in a Jupyter Notebook (`qr_auth.ipynb`), while `predict.py` is used for real-time QR code authentication.

## Features

- **Machine Learning & Deep Learning**: Uses both **Random Forest** and **CNN models** for classification.
- **Jupyter Notebook Pipeline**: `qr_auth.ipynb` contains the **data preprocessing, training, and evaluation**.
- **Standalone Prediction Script**: `predict.py` allows easy QR code classification using trained models.
- **Fast & Efficient**: Works with real-time QR code scans.

## Installation

### Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### Clone the Repository
```sh
git clone https://github.com/george-1-0-1/QR_Auth.git
cd QR_AUTH
```

### Set Up Virtual Environment (if neede)
```sh
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate  # For Windows
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Project Structure
```
QR_AUTH/
│── 📂 models/                  
│── 📂 QR Data/                
│── 📜 qr_auth.ipynb           
│── 📜 predict.py               
│── 📜 requirements.txt        
│── 📜 README.md                
```

## Training the Model

Run the `qr_auth.ipynb` notebook to train and save the models.

```sh
jupyter notebook qr_auth.ipynb
```

- The **Random Forest Model** will be saved as `random_forest_model.pkl`
- The **CNN Model** will be saved as `cnn_qr_model.h5`

## Predicting with a New QR Code

Once models are trained, use `predict.py` to classify QR codes.

```sh
python predict.py --image "path to sample_qr.png"
```

## Future Improvements

- **Data Augmentation**: Improve CNN generalization by adding image distortions.
- **Fine-tuning a Pretrained Model**: Use ResNet or EfficientNet for better feature extraction.
- **Mobile Deployment**: Convert the trained model to TensorFlow Lite for mobile authentication.

## Requirements

All dependencies are listed in `requirements.txt`. To install them, run:

```sh
pip install -r requirements.txt
```

### requirements.txt
```
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
joblib
```

## Troubleshooting

- **Model Not Found?** Ensure you've trained and saved the models (`random_forest_model.pkl` & `cnn_qr_model.h5`).
- **Incorrect Predictions?** Try retraining the CNN model with more **data augmentation**.
- **Image Not Loading in `predict.py`?** Double-check the **file path** and ensure the image exists.

## Contributing

Feel free to **fork this repository**, open an **issue**, or submit a **pull request** to improve this project.

## License

This project is licensed under the **MIT License**.
