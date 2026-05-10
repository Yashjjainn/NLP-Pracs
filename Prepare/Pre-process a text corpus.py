# Practical 1: Prepare/Pre-process a text corpus to make it more usable for NLP tasks using 
# tokenization, conversion to lowercase, removal of punctuation, filtration of stop 
# words, stemming and lemmatization.

import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocessing_pipeline():
    print("=== NLP Text Pre-processing Pipeline ===")
    
    # 1. Choice for User Input
    choice = input("Do you want to use the default text? (y/n): ").strip().lower()
    
    if choice == 'n':
        text = input("Enter the text you want to pre-process: ")
    else:
        text = "The quick brown foxes are jumping over the lazy dogs, while it's raining heavily!"
        print(f"\n[Using Default Text]: '{text}'")

    # 2. Conversion to lowercase & Tokenization
    # word_tokenize requires 'punkt_tab' to be downloaded
    tokens = word_tokenize(text.lower())

    # 3. Removal of punctuation
    # string.punctuation includes characters like !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    tokens_no_punct = [word for word in tokens if word not in string.punctuation]

    # 4. Filtration of stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens_no_punct if word not in stop_words]

    # 5. Stemming (Crude chopping)
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in filtered_tokens]

    # 6. Lemmatization (Linguistic root)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Displaying results
    print("\n--- Pre-processing Results ---")
    print(f"Original Text: {text}")
    print(f"\n1. Tokens (Cleaned): {tokens_no_punct}")
    print(f"2. After Stopword Removal: {filtered_tokens}")
    print(f"3. Stemmed Output: {stemmed}")
    print(f"4. Lemmatized Output: {lemmatized}")

if __name__ == "__main__":
    # Ensure these are downloaded in your Colab/Environment setup
    # nltk.download('punkt_tab')
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    preprocessing_pipeline()
