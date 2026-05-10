# Practical 9: Build and train a text classifier for the given data (using textblob or simpletransformers or keras library).

# ==========================================
# DEFAULT: TEXTBLOB (NAIVE BAYES) APPROACH
# ==========================================
# Run: pip install textblob
# Run: python -m textblob.download_corpora

from textblob.classifiers import NaiveBayesClassifier

def textblob_classifier_pipeline():
    print("=== Text Classifier using TextBlob (Naive Bayes) ===")
    
    train_data = [
        ("I absolutely loved this movie, it was fantastic!", "pos"),
        ("This is the best book I have ever read.", "pos"),
        ("What an amazing and joyful experience.", "pos"),
        ("Terrible plot, bad acting, and a waste of time.", "neg"),
        ("Worst movie I have ever seen.", "neg"),
        ("Boring, dull, and completely uninspired.", "neg")
    ]
    
    print("\nTraining the Naive Bayes model...")
    classifier = NaiveBayesClassifier(train_data)
    print("Model training complete!\n")
    
    while True:
        choice = input("Do you want to test a custom sentence? (y/n): ").strip().lower()
        if choice == 'n':
            test_text = "The acting was surprisingly good and I really enjoyed it."
            print(f"\n[Using Default Test Sentence]: '{test_text}'")
        else:
            test_text = input("Type your sentence to classify: ")

        prediction = classifier.classify(test_text)
        sentiment = "POSITIVE" if prediction == "pos" else "NEGATIVE"
        
        print("\n--- Prediction Results ---")
        print(f"Input: {test_text}")
        print(f"Sentiment: {sentiment}\n")
        
        if choice == 'n':
            break

if __name__ == "__main__":
    textblob_classifier_pipeline()


# ==========================================
# ALTERNATIVE 1: SIMPLETRANSFORMERS APPROACH
# ==========================================
"""
# Note: This requires heavy installations and downloading large pre-trained LLM weights.
# Run: pip install simpletransformers pandas torch

import pandas as pd
from simpletransformers.classification import ClassificationModel

def simpletransformers_pipeline():
    print("=== Text Classifier using SimpleTransformers (RoBERTa) ===")
    
    # SimpleTransformers expects a Pandas DataFrame with 'text' and 'labels' columns
    train_data = [
        ["I absolutely loved this movie, it was fantastic!", 1],
        ["This is the best book I have ever read.", 1],
        ["Terrible plot, bad acting, and a waste of time.", 0],
        ["Worst movie I have ever seen.", 0]
    ]
    train_df = pd.DataFrame(train_data, columns=["text", "labels"])
    
    # Initialize a pre-trained RoBERTa model
    # args dict configures it to train quickly for a demonstration
    model = ClassificationModel(
        "roberta", 
        "roberta-base", 
        args={"num_train_epochs": 1, "overwrite_output_dir": True, "use_multiprocessing": False}
    )
    
    # Fine-tune the LLM on our custom dataset
    print("Fine-tuning the Transformer model...")
    model.train_model(train_df)
    
    # Test the model
    test_sentence = ["The acting was surprisingly good and I really enjoyed it."]
    predictions, raw_outputs = model.predict(test_sentence)
    
    sentiment = "POSITIVE" if predictions[0] == 1 else "NEGATIVE"
    print(f"\nReview: {test_sentence[0]}")
    print(f"Predicted Sentiment: {sentiment}")

# To run this, uncomment the function call below:
# simpletransformers_pipeline()
"""


# ==========================================
# ALTERNATIVE 2: KERAS (DEEP LEARNING) APPROACH
# ==========================================
"""
# Note: This is a condensed version of Practical 8.
# Run: pip install tensorflow numpy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def keras_classifier_pipeline():
    print("=== Text Classifier using Keras (Neural Network) ===")
    
    reviews = ["I loved it, fantastic!", "Terrible and boring.", "Great movie.", "Worst ever."]
    labels = np.array([1, 0, 1, 0])
    
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(reviews)
    padded_seqs = pad_sequences(tokenizer.texts_to_sequences(reviews), maxlen=10)
    
    model = Sequential([
        Embedding(input_dim=100, output_dim=8, input_length=10),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Training Keras model...")
    model.fit(padded_seqs, labels, epochs=20, verbose=0)
    print("Training complete!")
    
    test_seq = pad_sequences(tokenizer.texts_to_sequences(["It was a great joy to watch."]), maxlen=10)
    prediction = model.predict(test_seq, verbose=0)[0][0]
    
    print(f"Prediction Confidence (Closer to 1 = Pos, Closer to 0 = Neg): {prediction:.4f}")

# To run this, uncomment the function call below:
# keras_classifier_pipeline()
"""
pass
