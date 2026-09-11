import re                 # To use regular expressions for text cleaning
import os                 # To handle file paths and directories
import kagglehub          # To download datasets from Kaggle
import pandas as pd       # To read and manipulate the dataset
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers import AveragePooling1D, BatchNormalization, Dense, Dropout, TextVectorization, GlobalMaxPooling1D
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras_tuner import RandomSearch
import seaborn as sns

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

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)

print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)

# Maximum number of words in the vocabulary
vocab_size = 17904         # Number of unique words in the cleaned review text column is 17,893, but I use 17904 because multiple of 8 is better for GPU processing

# Set padding length 
max_length = 131

# Create tokenizer
tokenizer = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=max_length
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
train_lengths = X_train.str.split().str.len()

print("Maximum sequence length:", train_lengths.max())
print("90th percentile:", np.percentile(train_lengths, 90))
print("95th percentile:", np.percentile(train_lengths, 95))
print("99th percentile:", np.percentile(train_lengths, 99))

num_truncated = (train_lengths > max_length).sum()

percentage_truncated = (
    num_truncated / len(train_lengths)
) * 100

print("Number of reviews > 131 tokens truncated:", num_truncated)
print("Percentage > 131 tokens truncated:", percentage_truncated, "%")

print("\nExample original review:")
print(X_train.iloc[0])

print("\nExample encoded & padded sequence:")
print(X_train_sequences[0])

# <//////////////////////////////////////-- Dataset Pre-processing --//////////////////////////////////////> 



# <//////////////////////////////////////-- Model Hyper Parameters Tuning --//////////////////////////////////////> 

# Bi-LSTM
def build_biLSTM_model(hp): 
    embedding_dim = 128

    model = Sequential()

    model.add(Input(shape=(max_length,)))

    model.add(
        Embedding(
            input_dim=actual_vocabulary_size,
            output_dim=embedding_dim
        )
    )

    model.add(Bidirectional(LSTM(
        units=hp.Int("lstm_units", min_value=32, max_value=256, step=32), 
        return_sequences=True, 
        activation="tanh",
        kernel_initializer=hp.Choice("kernel_initializer", ["he_normal", "truncated_normal", "orthogonal"]),
        kernel_regularizer=regularizers.l2(hp.Choice("kernel_regularizer_2", [0.0001, 0.0005, 0.001]))
        ))
    )

    model.add(Dropout(0.3))

    model.add(GlobalMaxPooling1D())

    # 1 sentiment class
    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model

tuner = RandomSearch(
    build_biLSTM_model,
    objective='val_accuracy',
    max_trials=1,
    executions_per_trial=1,
    directory='biLSTM_tuning_results',
    project_name='sentiment_analysis'
)

tuner.search(
    X_train_sequences,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val_sequences, y_val),
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters:")
print(best_hps.values)

model = tuner.hypermodel.build(best_hps)

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
axes[0].plot(epochs, val_acc, 'r', label='Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Training and Validation Accuracy')
axes[0].legend()

axes[1].plot(epochs, loss, 'b', label='Train Loss')
axes[1].plot(epochs, val_loss, 'r', label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training and Validation Loss')
axes[1].legend()

plt.tight_layout()
plt.show()

predictions = model.predict(X_test_sequences)
y_pred = (predictions >= 0.5).astype(int).flatten()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# <//////////////////////////////////////-- Model Hyper Parameters Tuning --//////////////////////////////////////> 