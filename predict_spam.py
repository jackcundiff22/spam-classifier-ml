import os
import joblib 

MODEL_PATH = "spam_pipeline.joblib"

def load_model(): 
    """Load the trained spam classification pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Train it first by running: python train_spam_classifier.py"
        )
    
    print(f"[INFO] Loading model from '{MODEL_PATH}'...")
    model = joblib.load(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
    return model 

def classify_message(model, message: str) -> str:
    """Return the predicted label ('spam' or 'ham') for one message."""
    prediction = model.predict([message])[0]
    return prediction 

def main():
    print("=== Spam Classifier ===")
    print("Type a message and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    model = load_model()

    while True:
        msg = input("Message: ")

        if msg.strip().lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        label = classify_message(model, msg)
        print(f"Prediction: {label}\n")

if __name__ == "__main__":
    main()