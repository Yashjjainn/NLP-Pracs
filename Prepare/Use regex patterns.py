# Practical 2: Use regex patterns to extract the usernames from the email addresses, 
# hashtags, dates, and phone numbers present in a given text.

import re

def regex_extraction_pipeline():
    print("=== Regex Information Extraction ===\n")
    
    # 1. Choice for User Input
    choice = input("Do you want to use the default text? (y/n): ").strip().lower()
    
    if choice == 'n':
        text = input("Enter the text you want to extract information from: ")
    else:
        text = """
        Hello, please contact admin_user123 at admin@university.edu for the 
        #NLP_Exam updates. The exam is scheduled for 15-05-2026. 
        For queries, call 9876543210 or reach out to support-team@gmail.com.
        Follow us for more: #AI #DeepLearning2026
        """
        print(f"\n[Using Default Text]: {text}")

    # 2. Regex Pattern Definitions
    # Usernames from emails: captures word characters before the @
    email_user_pattern = r'([a-zA-Z0-9_.+-]+)@'
    
    # Hashtags: # followed by word characters
    hashtag_pattern = r'#\w+'
    
    # Dates: DD-MM-YYYY format
    date_pattern = r'\d{2}-\d{2}-\d{4}'
    
    # Phone numbers: exactly 10 digits with boundaries
    phone_pattern = r'\b\d{10}\b'

    # 3. Extraction
    usernames = re.findall(email_user_pattern, text)
    hashtags = re.findall(hashtag_pattern, text)
    dates = re.findall(date_pattern, text)
    phones = re.findall(phone_pattern, text)

    # 4. Display Results
    print("\n--- Extraction Results ---")
    print(f"Extracted Usernames: {usernames}")
    print(f"Extracted Hashtags:  {hashtags}")
    print(f"Extracted Dates:     {dates}")
    print(f"Extracted Phone Nos: {phones}")

if __name__ == "__main__":
    regex_extraction_pipeline()
