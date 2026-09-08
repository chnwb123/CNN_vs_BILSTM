# Package imports
import re                 # To use regular expressions for text cleaning
import os                 # To handle file paths and directories
import kagglehub          # To download datasets from Kaggle
import pandas as pd       # To read and manipulate the dataset
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Download required NLTK data (run once only)
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")


# <//////////////////////////////////////-- Dataset Initialization --//////////////////////////////////////> 

# Download latest version of dataset
path = kagglehub.dataset_download("dongrelaxman/amazon-reviews-dataset")      # Amazon Reviews Dataset

# Get the path to the downloaded kaggle Amazon reviews dataset file 
print("Path to downloaded Amazon CSV dataset files:", path)

# Read the Amazon reviews dataset .CSV file 
amazon_dataset_path = os.path.join(path, "Amazon_Reviews.csv")

# Print the path to the Amazon reviews dataset .CSV file
print("\nPath to Amazon CSV dataset file:", amazon_dataset_path)    

# Transform the Amazon reviews dataset .CSV file into a pandas DataFrame
data = pd.read_csv(
    amazon_dataset_path,
    engine="python",
    # nrows=18500            # capped to 18,500 entries
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
# 2. Remove URL, punctuaition marks, stop words, special characters, numbers, and extra whitespace
# 3. Lemmatization
# 4. Text abbreaviation correction

def review_text_cleaning(review_text):
    # 1. Convert to lowercase
    text = review_text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 2. Tokenize into words
    tokens = word_tokenize(text)

    # Text abbreviation correction
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

    # Remove extra whitespace
    words = [word.strip() for word in words if word.strip()]

    # 3. Remove punctuation marks 
    table = str.maketrans("", "", string.punctuation)
    stripped = [w.translate(table) for w in words]
    words = [word for word in stripped if word.isalpha()]

    # concatenante "ca + not" to cant
    for i in range(len(words) - 1):
        if words[i] == "ca" and words[i + 1] == "not":
            words[i] = "cant"
            words[i + 1] = ""
        elif words[i] == "wo" and words[i + 1] == "not":
            words[i] = "wont"
            words[i + 1] = ""

    # 4. Remove stop words
    stop_words = set(stopwords.words("english"))
    words_to_keep = {"cant", "wont", "not", "no", "never", "nor", "very", "too", "really", "extremely", "highly"}
    custom_stopwords = stop_words - words_to_keep
    words = [word for word in words if word not in custom_stopwords]

    # 5. Lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # 6. Join the words back into a string
    cleaned_text = " ".join(word for word in words if word != "")

    return cleaned_text


# Function to tokenize, encode, and pad the cleaned review text
def tokenize_encode_pad(cleaned_review_text):
    return 

# Removed unnecessary columns from the Amazon reviews dataset
data_dropped_unnecessary_columns = data.drop(["Reviewer Name", "Profile Link", "Country", "Review Count", "Review Date", "Review Title", "Date of Experience"], axis=1)
data = data_dropped_unnecessary_columns

# Print some rows of the Amazon reviews dataset after deleting unnecessary columns
print("\nSome rows after deleting unnecessary columns:", data.head())

# Print the info of Amazon reviews dataset after deleting unnecessary columns
print("\nPrint dataset info after deleting unnecessary columns: ")
print(data.info())

# For Rating column, the rating value would be extracted and converted to an integer type
# "Rated 2 out of 5 stars" = 2 (int type)
data["Rating"] = data["Rating"].str.extract(r'(\d+)').astype(int)

# Print info and some rows of the Amazon reviews dataset after extracting rating values from the Rating column
print("\nDataset info after extracting rating values:")
print(data.info())
print("\nSome rows after extracting rating values from the Rating column:", data.head())

# Categorize ratings into sentiment classes
# 1 = Negative, 2 = Slightly Negative, 3 = Neutral, 4 = Slightly Positive, 5 = Positive
data["Sentiment"] = data["Rating"].apply(lambda x: "Negative" if x == 1 else ("Slightly Negative" if x == 2 else ("Neutral" if x == 3 else ("Slightly Positive" if x == 4 else "Positive"))))

# Print info and some rows of the Amazon reviews dataset after adding Sentiment column
print("\nDataset info after adding Sentiment column:")
print(data.info())
print("\nSome rows after adding Sentiment column:", data.head())
print("\nSentiment value counts:")
print(data["Sentiment"].value_counts())

# Print the average length of every review text in the Review Text column
data["Review Text Length"] = data["Review Text"].apply(len)
print("\nAverage length of review texts:", data["Review Text Length"].mean())

# Print the number of unique values in the Review Text column
print("\nNumber of unique values in the Review Text column:", data["Review Text"].nunique())

# Pass the Review Text column to the review_text_cleaning function to clean the review text
data["Cleaned Review Text"] = data["Review Text"].apply(review_text_cleaning)

print("\nSome rows after cleaning the Review Text column:", data.head())

# Print the average length of every review text in the Cleaned Review Text column
data["Cleaned Review Text Length"] = data["Cleaned Review Text"].apply(len)
print("\nAverage length of cleaned review texts:", data["Cleaned Review Text Length"].mean())

# Print the number of unique values in the Cleaned Review Text column
print("\nNumber of unique values in the Cleaned Review Text column:", data["Cleaned Review Text"].nunique())

data_in_CSV_format = data.to_csv(r'C:\APD3F2601SE\SEM_2\Courses\ODL\Assignment\Code\amazon_reviews_cleaned.csv', index=False)

# <//////////////////////////////////////-- Dataset Cleaning --//////////////////////////////////////> 


# 5. Tokenization
# 6. Encoding
# 7. Padding
# 8. Word Embedding

# Tokenize the cleaned review text into sequences of integers
# data["Tokenized Review Text"] = data["Cleaned Review Text"].apply(tokenize_encode_pad)

# Split the dataset into training and testing sets
# data_in_CSV_format = data.to_csv(r'C:\APD3F2601SE\SEM_2\Courses\ODL\Assignment\Code\amazon_reviews_cleaned.csv', index=False)

X = data["Cleaned Review Text"]
Y = data["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=2,
    stratify=Y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.125,
    random_state=2,
    stratify=y_train
)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)

# Maximum number of words in the vocabulary
vocab_size = 20000

# Create tokenizer
tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token="<OOV>"
)

# Fit tokenizer ONLY on training data
tokenizer.fit_on_texts(X_train)

# Convert text to integer sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_val_sequences = tokenizer.texts_to_sequences(X_val)

# Analyse sequence lengths
train_lengths = [len(sequence) for sequence in X_train_sequences]

print("\nMaximum sequence length:", max(train_lengths))
print("90th percentile:", np.percentile(train_lengths, 90))
print("95th percentile:", np.percentile(train_lengths, 95))
print("99th percentile:", np.percentile(train_lengths, 99))


# Padding
max_length = 128

num_truncated = sum(
    len(sequence) > max_length
    for sequence in X_train_sequences
)

percentage_truncated = (
    num_truncated / len(X_train_sequences)
) * 100

print("Number of truncated reviews:", num_truncated)
print("Percentage of truncated reviews:", percentage_truncated)

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_val_padded = pad_sequences(
    X_val_sequences,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

print("\nX_train padded shape:", X_train_padded.shape)
print("X_test padded shape:", X_test_padded.shape)
print("X_val padded shape:", X_val_padded.shape)

print("\nExample original review:")
print(X_train.iloc[0])

print("\nExample encoded sequence:")
print(X_train_sequences[0])

print("\nExample padded sequence:")
print(X_train_padded[0])


# <//////////////////////////////////////-- Model Building --//////////////////////////////////////> 
# <//////////////////////////////////////-- Model Building --//////////////////////////////////////> 
