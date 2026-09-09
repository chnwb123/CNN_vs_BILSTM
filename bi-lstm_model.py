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
from keras.layers import Input, Embedding, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers


data = pd.read_csv(r'C:\APD3F2601SE\SEM_2\Courses\ODL\Assignment\Code\amazon_reviews_cleaned.csv')

# <//////////////////////////////////////-- Dataset Cleaning --//////////////////////////////////////> 


# <//////////////////////////////////////-- Dataset Pre-processing --//////////////////////////////////////> 

# 1. Test and Train Split
# 2. Tokenization
# 3. Encoding
# 4. Padding
# 5. Word Embedding

# Split the dataset into training and testing sets
X = data["Cleaned Review Text"]
# Y = data["Rating"] - 1  # Subtracting 1 to make the ratings start from 0
Y = data["Rating"] 

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
actual_vocabulary_size = len(actual_vocabulary) + 1  # Adding 1 to account for the padding token

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

# <//////////////////////////////////////-- Dataset Pre-processing --//////////////////////////////////////> 



# <//////////////////////////////////////-- Model Building --//////////////////////////////////////> 

# Bi-LSTM
model = Sequential()

model.add(Input(shape=(max_length,)))

model.add(
    Embedding(
        input_dim=actual_vocabulary_size,
        output_dim=128
    )
)

model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))))

model.add(Dropout(0.5))

model.add(GlobalMaxPooling1D())

model.add(Dropout(0.5))

# Five sentiment classes
model.add(
    Dense(
        1,
        # activation="softmax"
        activation="sigmoid"
    )
)

model.compile(
    # loss="sparse_categorical_crossentropy",
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    X_train_sequences,
    y_train,
    batch_size=64,
    epochs=10,
    validation_data=(X_val_sequences, y_val),
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
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

# <//////////////////////////////////////-- Model Building --//////////////////////////////////////> 
