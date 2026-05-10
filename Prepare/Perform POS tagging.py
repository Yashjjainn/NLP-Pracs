# Practical 6: Perform POS tagging in a given text file. Extract all the nouns present in the text. Create and print a dictionary with frequency of parts of speech present in the document.

import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Note for Colab/Local Setup: 
# Ensure nltk.download('averaged_perceptron_tagger') and nltk.download('punkt') have been run!

def pos_tagging_pipeline():
    print("=== Part-of-Speech (POS) Tagging & Extraction ===")
    
    # 1. Ask the user for their preference
    choice = input("Do you want to use the default text? (y/n): ").strip().lower()
    
    if choice == 'n':
        text = input("Enter your text for POS tagging: ")
    else:
        text = "The diligent students are brilliantly studying Natural Language Processing for their final practical exam!"
        print(f"\n[Using Default Text]: '{text}'")

    # 2. Tokenize the text
    # POS tagging requires a list of tokens, not a raw string.
    tokens = word_tokenize(text)
    
    # 3. Perform POS Tagging
    # Returns a list of tuples: [('The', 'DT'), ('diligent', 'JJ'), ('students', 'NNS')...]
    pos_tags = nltk.pos_tag(tokens)
    
    # 4. Extract all Nouns
    # In the Penn Treebank tagset, all noun tags start with 'NN' (NN, NNS, NNP, NNPS)
    nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
    
    # 5. Create a dictionary with the frequency of parts of speech
    # We extract just the tag from each (word, tag) tuple and count them
    tag_frequencies = dict(Counter(tag for word, tag in pos_tags))
    
    # Print the results clearly
    print("\n--- 1. Full POS Tags ---")
    print(pos_tags)
    
    print(f"\n--- 2. Extracted Nouns (Total: {len(nouns)}) ---")
    print(nouns)
    
    print("\n--- 3. POS Frequency Dictionary ---")
    # Formatting the dictionary nicely for the console
    for tag, freq in tag_frequencies.items():
        print(f"{tag}: {freq}")

# Run the function
if __name__ == "__main__":
    pos_tagging_pipeline()
