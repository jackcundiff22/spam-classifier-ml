# Spam Message Classifier (Machine Learning Project)

A machine-learning powered SMS Spam Detector built with **Python**, **Scikit-Learn**, and **TF-IDF text vectorization**.  
This project loads a labeled dataset of text messages, trains a Naive Bayes classifier, evaluates the model, and allows users to interactively test messages for spam.

---

#  Author

**Jack Cundiff**  
Computer Science | AI & Software Developer  
**GitHub:** https://github.com/jackcundiff22

##  Features

-  Loads and cleans SMS spam dataset (`spam.csv`)
-  Converts text into numerical features using **TF-IDF**
-  Trains a **Multinomial Naive Bayes** classification model
-  Provides accuracy, confusion matrix, and full classification report
-  Saves trained model as `spam_pipeline.joblib`
-  Interactive CLI prediction script (`predict_spam.py`)
-  Works with any dataset using columns:
  - `v1` (`spam` / `ham`)
  - `v2` (message text)

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Programming language |
| **Pandas** | Data loading & cleaning |
| **Scikit-Learn** | TF-IDF, Naive Bayes, accuracy metrics |
| **Joblib** | Model serialization |
| **Virtual Environment (venv)** | Dependency isolation |

---

##  Project Structure
spam-classifier-ml/
│
├── train_spam_classifier.py # Train the model
├── predict_spam.py # Interactive predictor
├── spam.csv # Dataset (Kaggle SMS Spam Collection)
├── spam_pipeline.joblib # Saved ML model (generated after training)
├── venv/ # Virtual environment
└── README.md # Project documentation

# Installation & Setup

## 1. Clone the repository
```
git clone https://github.com/<your-username>/spam-classifier-ml.git
cd spam-classifier-ml
```
##  2. Create and activate virtual environment
```
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
```
## 3. Install dependencies
```
pip install pandas scikit-learn joblib
```
## Dataset

This project uses the SMS Spam Collection Dataset (Kaggle), containing real labeled text messages.

Column: v1, v2	  


Meaning: Label (spam or ham),  Message text


Replace or update spam.csv with any compatible dataset.

## Train the Model

Run:
```
python train_spam_classifier.py
```

This script:

Loads dataset

Cleans + prepares data

Trains Naive Bayes model

Prints accuracy + metrics

Saves pipeline to spam_pipeline.joblib

## Predict Spam Messages (CLI)

After training:
```
python predict_spam.py
```

Example:
```
=== Spam Classifier ===
Type a message and press Enter.
Type 'quit' or 'exit' to stop.

Message: You won a free iPhone! Click here to claim.
Prediction: spam
```
## Example Model Performance

Typical accuracy (varies by dataset):
```
Accuracy: 0.9652

Classification Report:
              precision    recall  f1-score   support
ham              0.98        0.98       0.98      XXXX
spam             0.93        0.92       0.92      XXXX

Confusion Matrix:
[[TN  FP]
 [FN  TP]]
```
## Future Improvements 

- Add spam probability scoring (predict_proba)

- Build a Tkinter GUI version

- Deploy as a Flask API

- Add model comparison (SVM, Logistic Regression)

- Visualize dataset distribution + word clouds


