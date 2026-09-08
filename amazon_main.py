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
from keras.layers import AveragePooling1D, BatchNormalization, Dense, Dropout, TextVectorization, GlobalMaxPooling1D
from keras.models import Sequential
from keras.layers import Input, Embedding, Conv1D
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.utils import to_categorical



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
    nrows=5000          # capped to 18,500 entries
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


# <//////////////////////////////////////-- Dataset Pre-processing --//////////////////////////////////////> 

# 1. Test and Train Split
# 2. Tokenization
# 3. Encoding
# 4. Padding
# 5. Word Embedding

# Split the dataset into training and testing sets
X = data["Cleaned Review Text"]
Y = data["Rating"] - 1  # Subtracting 1 to make the ratings start from 0

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

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)

# Maximum number of words in the vocabulary
vocab_size = 17904         # Number of unique words in the cleaned review text column is 17,893, but I use 17904 because multiple of 8 is better for GPU processing

# Create tokenizer
tokenizer = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=128
)

# Fit tokenizer ONLY on training data
tokenizer.adapt(X_train)

actual_vocabulary = tokenizer.get_vocabulary()
actual_vocabulary_size = len(actual_vocabulary)

print("Actual vocabulary size:", actual_vocabulary_size)

# Convert text to integer sequences
X_train_sequences = tokenizer(X_train)
X_test_sequences = tokenizer(X_test)
X_val_sequences = tokenizer(X_val)

print("\nX_train sequences shape:", X_train_sequences.shape)
print("X_test sequences shape:", X_test_sequences.shape)
print("X_val sequences shape:", X_val_sequences.shape)


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

print("\nExample original review:")
print(X_train.iloc[0])

print("\nExample encoded & padded sequence:")
print(X_train_sequences[0])

# embedding_layer = Embedding(
#     input_dim=vocab_size,
#     output_dim=128,
#     input_shape=(max_length,)
# )

# <//////////////////////////////////////-- Dataset Pre-processing --//////////////////////////////////////> 



# <//////////////////////////////////////-- Model Building --//////////////////////////////////////> 

# model = Sequential([
#     Input(shape=(10,)),                                           #  Explicit Input layer
#     Embedding(input_dim=1000, output_dim=64),                     #  Remove input_shape here
#     # ... other layers
# ])

# CNN
model = Sequential()

model.add(Input(shape=(max_length,)))

model.add(
    Embedding(
        input_dim=actual_vocabulary_size,
        output_dim=128
    )
)

model.add(
    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    )
)

model.add(BatchNormalization())

model.add(
    AveragePooling1D(
        pool_size=2
    )
)

model.add(Dropout(0.5))

model.add(
    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    )
)

model.add(BatchNormalization())

model.add(
    AveragePooling1D(
        pool_size=2
    )
)

model.add(Dropout(0.5))

# Convert sequence representation into one vector per review
model.add(GlobalMaxPooling1D())

# Five sentiment classes
model.add(
    Dense(
        5,
        activation="softmax"
    )
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    X_train_sequences,
    y_train,
    batch_size=10,
    epochs=10,
    validation_data=(X_val_sequences, y_val)
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
# Creating subplots, figsize (width, height) in inches
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
axes[0].plot(epochs, acc, 'b', label='Train Accuracy')
axes[0].plot(epochs, val_acc, 'r', label='Test Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Training and Validation Accuracy')
axes[0].legend()

axes[1].plot(epochs, loss, 'b', label='Train Loss')
axes[1].plot(epochs, val_loss, 'r', label='Test Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training and Validation Loss')
axes[1].legend()

plt.tight_layout()
plt.show()



# Bi-LSTM




# <//////////////////////////////////////-- Model Building --//////////////////////////////////////> 
