# Practical 7: Identify and print the named entities using Name Entity Recognition (NER) for a collection of news headlines.

import spacy

# Note for Colab/Local Setup: 
# Ensure you have run: python -m spacy download en_core_web_sm

def ner_pipeline():
    print("=== Named Entity Recognition (NER) with spaCy ===")
    
    # 1. Load the pre-trained NLP model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("\n[ERROR]: The spaCy model 'en_core_web_sm' is not installed.")
        print("Please run: '!python -m spacy download en_core_web_sm' in your terminal or Colab cell first.")
        return

    # 2. Ask the user for their preference
    choice = input("Do you want to use the default news headlines? (y/n): ").strip().lower()
    
    if choice == 'n':
        text = input("Enter a news headline for NER extraction: ")
    else:
        # Default text containing multiple entities (Money, GPE, ORG, Person, Date)
        text = "Apple is looking at buying a U.K. startup for $1 billion. Meanwhile, Google's CEO Sundar Pichai announced a new AI center in New Delhi starting January 2027."
        print(f"\n[Using Default Headlines]:\n'{text}'")

    # 3. Process the text through the spaCy pipeline
    doc = nlp(text)
    
    # 4. Extract and print the Named Entities
    print("\n--- Extracted Named Entities ---")
    
    if not doc.ents:
        print("No named entities found in the given text.")
    else:
        # Formatting the output nicely
        print(f"{'ENTITY':<20} | {'LABEL'}")
        print("-" * 35)
        for ent in doc.ents:
            # ent.text gives the actual word, ent.label_ gives the category
            print(f"{ent.text:<20} | {ent.label_}")
            
    # 5. Quick guide for the examiner
    print("\n--- Quick Label Guide ---")
    print("ORG    : Companies, agencies, institutions.")
    print("PERSON : People, including fictional.")
    print("GPE    : Geopolitical entities (Countries, cities, states).")
    print("MONEY  : Monetary values, including unit.")
    print("DATE   : Absolute or relative dates or periods.")

# Run the function
if __name__ == "__main__":
    ner_pipeline()
