# Practical 3: List the most common words (with their frequency) in a given text excluding stopwords. 

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

def common_words_pipeline():
    print("=== Most Common Words Extractor (Excluding Stopwords) ===\n")
    
    # 1. Choice for User Input
    choice = input("Do you want to use the default text? (y/n): ").strip().lower()
    
    if choice == 'n':
        text = input("Enter the text you want to analyze: ")
    else:
        # Default text containing several repeating technical terms [cite: 1, 11]
        text = """
        Natural Language Processing (NLP) is a subfield of artificial intelligence. 
        NLP techniques are used to help computers understand human language. 
        In NLP, we use Python because Python libraries make Natural Language Processing easier.
        """
        print(f"\n[Using Default Text]: {text}")

    # 2. Tokenization and Lowercasing 
    # We use .isalpha() to ensure we only count actual words, not punctuation 
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]

    # 3. Stopword Filtration 
    # Loading the standard English stopword list [cite: 1, 21]
    stop_words = set(stopwords.words('english'))
    meaningful_words = [word for word in tokens if word not in stop_words]

    # 4. Count Frequencies 
    word_counts = Counter(meaningful_words)

    # 5. Extract Top 5 Results 
    top_n = 5
    most_common = word_counts.most_common(top_n)

    # 6. Display Results
    print(f"\n--- Top {top_n} Most Common Meaningful Words ---")
    print(f"{'WORD':<15} | {'FREQUENCY'}")
    print("-" * 30)
    for word, count in most_common:
        print(f"{word:<15} | {count}")

if __name__ == "__main__":
    # Ensure necessary NLTK data is available [cite: 1, 10, 11]
    # nltk.download('punkt')
    # nltk.download('punkt_tab')
    # nltk.download('stopwords')
    common_words_pipeline()
