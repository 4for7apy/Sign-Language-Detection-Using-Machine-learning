# Sign Language Recognition using Random Forest

## 📌 Overview
This project aims to recognize American Sign Language (ASL) hand signs using a **Random Forest** classifier. The model is trained on grayscale images of hand signs representing alphabets (A-Z) and numbers (1-9).

## 🔍 Features
- **Real-time ASL recognition** using a webcam
- **Preprocessed grayscale images** for training
- **Random Forest classifier** for sign recognition
- **User-friendly interface** to predict and display signs

## 📂 Dataset
The dataset consists of:
- Hand signs labeled from **A-Z** and **1-9**
- Images converted to **grayscale** for better feature extraction
- Augmented data to improve model performance

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Libraries Used:** OpenCV, Scikit-learn, NumPy, Pandas, Matplotlib
- **Model:** Random Forest Classifier

## 🚀 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sign-language-recognition.git
   cd sign-language-recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## 🏋️ Training the Model
To train the model, use the following command:
```bash
python train.py
```
This will preprocess the dataset and train the **Random Forest** classifier.

## 📊 Results
- Achieved **high accuracy** on test data
- Works well with real-time sign detection
