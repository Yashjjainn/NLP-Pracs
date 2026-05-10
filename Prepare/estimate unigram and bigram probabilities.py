# Practical 5: Build a simple statistical language model: estimate unigram and bigram probabilities with add-one smoothing and compute the probability of given sentences.

import nltk
from collections import Counter
from nltk.tokenize import word_tokenize

def calculate_language_model():
    print("=== N-Gram Language Model with Add-One Smoothing ===")
    
    # 1. Ask the user for their preference
    choice = input("Do you want to use the default text? (y/n): ").strip().lower()
    
    if choice == 'n':
        corpus_text = input("Enter your training corpus (e.g., a few sentences): ")
        test_sentence = input("Enter a sentence to calculate its probability: ")
    else:
        corpus_text = "I love natural language processing. I love machine learning. Machine learning is fun."
        test_sentence = "I love learning."
        print(f"\n[Using Default Corpus]: '{corpus_text}'")
        print(f"[Using Default Test Sentence]: '{test_sentence}'")

    # 2. Tokenize the training corpus (lowercase, alphabetic only)
    corpus_tokens = [w.lower() for w in word_tokenize(corpus_text) if w.isalpha()]
    
    # 3. Calculate Unigram & Bigram frequencies, and Vocabulary size (V)
    unigram_counts = Counter(corpus_tokens)
    bigram_counts = Counter(nltk.bigrams(corpus_tokens))
    vocab_size = len(set(corpus_tokens)) # Unique words (V)
    
    print(f"\nVocabulary Size (V): {vocab_size}")

    # 4. Tokenize the test sentence and generate its bigrams
    test_tokens = [w.lower() for w in word_tokenize(test_sentence) if w.isalpha()]
    test_bigrams = list(nltk.bigrams(test_tokens))
    
    # 5. Calculate probabilities using Add-One (Laplace) Smoothing
    sentence_probability = 1.0
    
    print("\n--- Bigram Probabilities ---")
    for w1, w2 in test_bigrams:
        # Formula: (Count(w1, w2) + 1) / (Count(w1) + V)
        bigram_prob = (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + vocab_size)
        
        # Multiply to the total sentence probability
        sentence_probability *= bigram_prob
        
        print(f"P('{w2}' | '{w1}') = {bigram_prob:.4f}")

    # Output the final result using scientific notation (since probabilities get very small)
    print(f"\nTotal Probability of '{test_sentence}': {sentence_probability:.8e}")

# Run the function
if __name__ == "__main__":
    calculate_language_model()
