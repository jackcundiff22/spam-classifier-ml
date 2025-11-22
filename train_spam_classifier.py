import os 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib 

DATA_FILE = 'spam.csv'
MODEL_PATH = "spam_pipeline.joblib"

def load_data(): 
    """Loads spam dataset and returns text + labels."""

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset '{DATA_FILE}' not found.")
   
    df = pd.read_csv(DATA_FILE, encoding="latin-1")

    # Common dataset column names
    possible_label_cols = ["label", "Category", "v1"]
    possible_text_cols = ["text", "Message", "v2"]

    label_col = None
    text_col = None

    for col in df.columns: 
        if col in possible_label_cols: 
            label_col = col
        if col in possible_text_cols:
            text_col = col 
    
    if label_col is None or text_col is None: 
        print("[DEBUG] Columns found:", df.columns)
        raise ValueError(
            "Could not detect label/text column names. "
            "Update the column names in load_data()."
        )
    
    df = df[[label_col, text_col]].dropna()

    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    print(f"[INFO] Loaded {len(df)} messages.")
    print("[INFO] Label distribution:")
    print(y.value_counts())

    return X, y

def build_pipeline(): 
    """Creates a text classification pipeline: TF-IDF vectorizer + Naive Bayes classifier"""

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=2
        )),
        ("clf", MultinomialNB())
    ])

    return pipeline 

def train_and_evaluate():
    """Loads data, trains the model, evaluates it, and saves the trained pipeline."""

    # 1) Load dataset 
    X, y = load_data()
    
    # 2) Split into train and test sets 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("[INFO] Training model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 3) Evaluate on test set 
    print("[INFO] Evaluating model...")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Accuracy: {accuracy: .4f}\n")

    print("[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred))

    print("[RESULT] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 4) Save the trained model 
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Model saved to '{MODEL_PATH}'")


if __name__ == "__main__": 
    train_and_evaluate()