# <//////////////////////////////////////-- Packages Import --//////////////////////////////////////> 

import re                 # To use regular expressions for text cleaning
import os                 # To handle file paths and directories
import kagglehub          # To download datasets from Kaggle
import pandas as pd       # To read and manipulate the dataset
import string             # To handle string operations
import nltk               # To use Natural Language Toolkit for text processing 
from nltk.corpus import stopwords           # To remove common words that do not contribute to the meaning of the text
from nltk.tokenize import word_tokenize     # To split text into individual words

# <//////////////////////////////////////-- Packages Import --//////////////////////////////////////> 


# Download required NLTK data (run once only)
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")


# <//////////////////////////////////////-- Dataset Initialization --//////////////////////////////////////> 

# Download Amazon reviews dataset from Kaggle using the kagglehub package 
path = kagglehub.dataset_download("dongrelaxman/amazon-reviews-dataset")     

# Print the path to the downloaded kaggle Amazon reviews dataset file 
print("\nPath to downloaded Amazon CSV dataset files:", path)

# Get the path to the downloaded kaggle Amazon reviews dataset .CSV file 
amazon_dataset_path = os.path.join(path, "Amazon_Reviews.csv")

# Print the path to the downloaded kaggle Amazon reviews dataset .CSV file 
print("\nPath to Amazon CSV dataset file:", amazon_dataset_path)    

# Transform the Amazon reviews dataset .CSV file into a pandas DataFrame
data = pd.read_csv(
    amazon_dataset_path,
    engine="python",
    nrows=18500            # capped to 18,500 entries
)

# Print some rows of the Amazon reviews dataset
print("\nSome rows of the amazon dataset: \n", data.head())

# Print the info of Amazon reviews dataset 
print("\nPrint dataset info: ")
print(data.info())

# <//////////////////////////////////////-- Dataset Initialization --//////////////////////////////////////> 




# <//////////////////////////////////////-- Dataset Cleaning --//////////////////////////////////////> 


# Function to clean the review text

# 1. Case folding
# 2. Remove URL
# 3. Tokenization
# 4. Text abbreaviation correction
# 5. Extra whitespace, punctuation marks, stop words, special characters, and numbers removal
# 6. Lemmatization


def review_text_cleaning(review_text):
    # 1. Convert to lowercase
    text = review_text.lower()

    # 2. Remove URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Tokenize the review text into individual words
    tokens = word_tokenize(text)

    # 4. Correct text abbreviation format
    abbreviations = {
        "n't": "not",
        "'re": "are",
        "'s": "is",
        "'d": "would",
        "'ll": "will",
        "'t": "not",
        "'ve": "have",
        "'m": "am"
    }
    words = [abbreviations.get(word, word) for word in tokens]

    # 5. Remove extra whitespace
    words = [word.strip() for word in words if word.strip()]

    # 6. Remove punctuation marks 
    table = str.maketrans("", "", string.punctuation)
    stripped = [w.translate(table) for w in words]
    words = [word for word in stripped if word.isalpha()]

    # 7. Concatenate separated words resulted from text abbreviation correction. E.g. can't -> ca not
    for i in range(len(words) - 1):
        if words[i] == "ca" and words[i + 1] == "not":
            words[i] = "cant"
            words[i + 1] = ""
        elif words[i] == "wo" and words[i + 1] == "not":
            words[i] = "wont"
            words[i + 1] = ""

    # 8. Remove stop words
    stop_words = set(stopwords.words("english"))
    words_to_keep = {"cant", "wont", "not", "no", "never", "nor", "very", 
                     "too", "really", "extremely", "highly"}
    custom_stopwords = stop_words - words_to_keep
    words = [word for word in words if word not in custom_stopwords]

    # 9. Lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # 10. Join the words back into a string
    cleaned_text = " ".join(word for word in words if word != "")

    # 11. Return the grouped words as a single string
    return cleaned_text


# Remove unnecessary columns from the Amazon reviews dataset
data_dropped_unnecessary_columns = data.drop(["Reviewer Name", 
                                              "Profile Link", 
                                              "Country", 
                                              "Review Count", 
                                              "Review Date", 
                                              "Review Title", 
                                              "Date of Experience"], axis=1)
data = data_dropped_unnecessary_columns

# Print some rows of the Amazon reviews dataset after deleting unnecessary columns
print("\nSome rows after deleting unnecessary columns: \n", data.head())

# Print the info of Amazon reviews dataset after deleting unnecessary columns
print("\nPrint dataset info after deleting unnecessary columns: ")
print(data.info())

# For Rating column, the rating value would be extracted and converted into an integer type
# "Rated 2 out of 5 stars" = 2 (int type)
data["Rating"] = data["Rating"].str.extract(r'(\d+)').astype(int)

# Change Rating values to 0 for 1, 2, 3 and 1 for 4, 5
data["Rating"] = data["Rating"].apply(lambda x: 0 if x == 1 or x == 2 or x == 3 else (1))

# Print some rows of the Amazon reviews dataset after extracting rating values from the Rating column and converting each of them into 0 or 1 integer value
print("\nSome rows after extracting and formatting rating values from the Rating column: \n", data.head())

# Print the info of the Amazon reviews dataset after extracting rating values from the Rating column and converting each of them into 0 or 1 integer value
print("\nDataset info after extracting and formatting rating values:")
print(data.info())

# Categorize Rating values into sentiment classes
# 0 = Negative sentiment (1, 2, 3 stars)
# 1 = Positive sentiment (4, 5 stars)
data["Sentiment"] = data["Rating"].apply(lambda x: "Postive" if x == 1 else ("Negative"))

# Print some rows of the Amazon reviews dataset after adding Sentiment column
print("\nSome rows after adding Sentiment column: \n", data.head())

# Print the info of the Amazon reviews dataset after adding Sentiment column
print("\nDataset info after adding Sentiment column:")
print(data.info())

# Print the count of each sentiment class in the Sentiment column. E.g. Postive: 10,000, Negative: 8,500
print("\nSentiment value counts:")
print(data["Sentiment"].value_counts())

# Get the length of every review text in the Review Text column
data["Review Text Length"] = data["Review Text"].apply(len)

# Print the average length of every review text in the Review Text column
print("\nAverage length of review texts:", data["Review Text Length"].mean())

# Print the number of unique values in the Review Text column
print("\nNumber of unique values in the Review Text column:", data["Review Text"].nunique())

# Pass the Review Text column to the review_text_cleaning function to clean the review text
data["Cleaned Review Text"] = data["Review Text"].apply(review_text_cleaning)

# Print some rows of the Amazon reviews dataset after cleaning the Review Text column
print("\nSome rows after cleaning the Review Text column: \n", data.head())

# Get the length of every cleaned review text in the Cleaned Review Text column
data["Cleaned Review Text Length"] = data["Cleaned Review Text"].apply(len)

# Print the average length of every cleaned review text in the Cleaned Review Text column
print("\nAverage length of cleaned review texts:", data["Cleaned Review Text Length"].mean())

# Print the number of unique values in the Cleaned Review Text column
print("\nNumber of unique values in the Cleaned Review Text column:", data["Cleaned Review Text"].nunique())

# Save the cleaned Amazon reviews dataset into a new .CSV file called "amazon_reviews_cleaned.csv" in the current working directory
data_in_CSV_format = data.to_csv(r'C:\APD3F2601SE\SEM_2\Courses\ODL\Assignment\Code\amazon_reviews_cleaned.csv', index=False)

# Print sucess message after saving the cleaned Amazon reviews dataset 
print("\nCleaned Amazon reviews dataset saved as 'amazon_reviews_cleaned.csv' in the current working directory.")

# <//////////////////////////////////////-- Dataset Cleaning --//////////////////////////////////////> 