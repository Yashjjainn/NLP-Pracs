# Practical 4: Create the TF-IDF (Term Frequency -Inverse Document Frequency) Matrix 
# for the given set of text documents. [cite: 22]

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def tfidf_pipeline():
    print("=== TF-IDF Matrix Generator ===\n")
    
    # 1. Choice for User Input
    choice = input("Do you want to use the default documents? (y/n): ").strip().lower()
    
    if choice == 'n':
        print("Enter 3 documents (sentences) one by one:")
        doc1 = input("Doc 1: ")
        doc2 = input("Doc 2: ")
        doc3 = input("Doc 3: ")
        documents = [doc1, doc2, doc3]
    else:
        # Default corpus
        documents = [
            "Machine learning is fascinating.",
            "Natural language processing is a subset of machine learning.",
            "Deep learning models excel at natural language processing."
        ]
        print("\n[Using Default Documents]:")
        for i, doc in enumerate(documents):
            print(f"Doc {i+1}: {doc}")

    # 2. Initialize the TF-IDF Vectorizer 
    # This handles tokenization and lowercase conversion automatically [cite: 1]
    vectorizer = TfidfVectorizer()

    # 3. Fit and Transform the documents into the TF-IDF matrix 
    tfidf_matrix = vectorizer.fit_transform(documents)

    # 4. Convert the matrix to a Pandas DataFrame for better visibility
    # get_feature_names_out() retrieves the vocabulary words 
    df_tfidf = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=vectorizer.get_feature_names_out(),
        index=['Doc 1', 'Doc 2', 'Doc 3']
    )

    # 5. Display Results
    print("\n--- Resulting TF-IDF Matrix ---")
    # Using to_string() ensures the table doesn't get truncated in the console
    print(df_tfidf.to_string())
    
    print("\nNote: Higher values indicate words that are more unique/important to that specific document.")

if __name__ == "__main__":
    tfidf_pipeline()
