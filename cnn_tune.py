# <//////////////////////////////////////-- Packages Import --//////////////////////////////////////> 

import pandas as pd       # To read and manipulate the dataset
import numpy as np        # To handle numerical operations

from sklearn.model_selection import train_test_split     # To split the dataset into training and testing sets
from sklearn.metrics import classification_report        # To generate a classification report that includes precision, recall, f1-score, and support for each class
from sklearn.metrics import confusion_matrix             # To generate a confusion matrix that shows the number of true positive, true negative, false positive, and false negative predictions

# <--- Keras imports for building and training the neural network model --->
from keras.models import Sequential             # To create a linear stack of layers for the neural network model

from keras.layers import AveragePooling1D       # To downsample the input representation by taking the average over a fixed window
from keras.layers import BatchNormalization     # To normalize the activations of the previous layer at each batch, i.e., applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1
from keras.layers import Dense                  # To add a fully connected layer to the neural network
from keras.layers import Dropout                # To prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time
from keras.layers import TextVectorization      # To convert a batch of strings (one sample = one string) into either a list of token indices (one sample = 1D tensor of integer token indices) or a dense representation (one sample = 1D tensor of float values)
from keras.layers import GlobalMaxPooling1D     # To downsample the input representation by taking the maximum value over a fixed window
from keras.layers import Input                  # To instantiate a Keras tensor
from keras.layers import Embedding              # To turn positive integers (indexes) into dense vectors of fixed size
from keras.layers import Conv1D                 # To create a convolutional layer for 1D inputs (e.g., temporal data)

from keras.callbacks import EarlyStopping       # To stop training when a monitored metric has stopped improving
from keras import regularizers                  # To apply penalties on layer parameters or layer activity during optimization

# <--- Keras imports for building and training the neural network model --->

from keras_tuner import RandomSearch            # To perform hyperparameter tuning using random search 

import matplotlib.pyplot as plt                 # To create visualizations and plots
import seaborn as sns                           # To create visualizations and plots, especially for statistical graphics

# <//////////////////////////////////////-- Packages Import --//////////////////////////////////////> 


# Read the cleaned Amazon reviews dataset from a CSV file into a pandas DataFrame
data = pd.read_csv(r'C:\APD3F2601SE\SEM_2\Courses\ODL\Assignment\Code\amazon_reviews_cleaned.csv')


# <//////////////////////////////////////-- Dataset Pre-processing --//////////////////////////////////////> 

# 1. Test and Train Split
# 2. Tokenization
# 3. Encoding
# 4. Padding

# Split the dataset into training and testing sets
X = data["Cleaned Review Text"]       # Independent variable (features) - the cleaned review text
Y = data["Rating"]                    # Dependent variable (target) - the rating (0 or 1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.20,                   # 20% of the data will be used for testing, and 80% for training
    random_state=2,                   # A random seed to ensure reproducibility of the results
    stratify=Y                        # To ensure that the proportion of classes in the training and testing sets is the same as in the original dataset
)

# Split the training set into training and validation sets 
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.125,                  # 12.5% of the training data will be used for validation, and 87.5% for training
    random_state=2,
    stratify=y_train                  # To ensure that the proportion of classes in the training and validation sets is the same as in the original training set
)

# Print the shapes of the training, testing, and validation sets
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)

print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)

# Maximum number of words in the vocabulary
vocab_size = 17904         # Number of unique words in the cleaned review text column is 17,893, but I use 17904 because multiple of 8 is better for GPU processing

# Set padding length 
max_length = 131           # The maximum length of the review text sequences after tokenization and padding. Any review text longer than this will be truncated, and any review text shorter than this will be padded with zeros.

# Create tokenizer
tokenizer = TextVectorization(
    max_tokens=vocab_size,                  # The maximum number of unique tokens (words) to keep in the vocabulary. Any words beyond this limit will be ignored.
    output_mode="int",                      # Each word will be represented by a unique integer index.
    output_sequence_length=max_length       # The length of the output sequences after tokenization.        
)

# Fit tokenizer only on training data
tokenizer.adapt(X_train)

# Get the actual vocabulary and its size
actual_vocabulary = tokenizer.get_vocabulary()
actual_vocabulary_size = len(actual_vocabulary)

# Print the actual vocabulary size
print("Actual vocabulary size:", actual_vocabulary_size)

# Convert training, testing, and validation texts to integer sequences
X_train_sequences = tokenizer(X_train)
X_test_sequences = tokenizer(X_test)
X_val_sequences = tokenizer(X_val)

# Print the shapes of the integer sequences for training, testing, and validation sets
print("\nX_train sequences shape:", X_train_sequences.shape)
print("X_test sequences shape:", X_test_sequences.shape)
print("X_val sequences shape:", X_val_sequences.shape)

# Analyse sequence lengths
train_lengths = X_train.str.split().str.len()

print("Maximum sequence length:", train_lengths.max())          # Print the maximum length of the review text sequences in the training set
print("90th percentile:", np.percentile(train_lengths, 90))     # Print the 90th percentile of the review text sequence lengths in the training set
print("95th percentile:", np.percentile(train_lengths, 95))     # Print the 95th percentile of the review text sequence lengths in the training set
print("99th percentile:", np.percentile(train_lengths, 99))     # Print the 99th percentile of the review text sequence lengths in the training set

# Count the number of review texts in the training set that are longer than the maximum length (131 tokens) and will be truncated
num_truncated = (train_lengths > max_length).sum()

percentage_truncated = (
    num_truncated / len(train_lengths)
) * 100

# Print the number and percentage of review texts in the training set that are longer than the maximum length (131 tokens) and will be truncated
print("Number of reviews > 131 tokens truncated:", num_truncated)
print("Percentage > 131 tokens truncated:", percentage_truncated, "%")

# Print an example of an original review text from the training set 
print("\nExample original review:")
print(X_train.iloc[0])

# Print an example of an encoded and padded sequence from the training set
print("\nExample encoded & padded sequence:")
print(X_train_sequences[0])

# <//////////////////////////////////////-- Dataset Pre-processing --//////////////////////////////////////> 


# <//////////////////////////////////////-- Model Hyper Parameters Tuning --//////////////////////////////////////> 

# Function to build the CNN model with hyperparameters to be tuned
def build_cnn_model(hp):
    embedding_dim = 128

    model = Sequential()

    model.add(Input(shape=(max_length,)))

    model.add(
        Embedding(
            input_dim=actual_vocabulary_size,
            output_dim=embedding_dim
        )
    )

    model.add(
        Conv1D(
            filters=hp.Int("filters_1", min_value=32, max_value=256, step=32),   # The range of output filters that would be tuned from 32 to 256, with a step size of 32.
            kernel_size=hp.Int("kernel_size_1", min_value=3, max_value=7, step=1),  # The range of kernel sizes that would be tuned from 3 to 7, with a step size of 1.
            activation="relu",
            kernel_initializer='he_normal', 
            kernel_regularizer=regularizers.l2(hp.Choice("kernel_regularizer_2", [0.0001, 0.0005, 0.001]))  # The range of kernel regularization values that would be tuned from 0.0001 to 0.001, with a step size of 0.0005.
        )
    )

    model.add(BatchNormalization())

    model.add(
        AveragePooling1D(
            pool_size=hp.Int("pool_size_1", [2, 3, 4])  # The range of pool sizes that would be tuned from 2 to 4, with a step size of 1.
        )
    )

    model.add(Dropout(hp.Float("dropout_1", min_value=0.2, max_value=0.5, step=0.1)))  # The range of dropout rates that would be tuned from 0.2 to 0.5, with a step size of 0.1.

    model.add(
        Conv1D(
            filters=hp.Int("filters_2", min_value=32, max_value=256, step=32),   
            kernel_size=hp.Int("kernel_size_2", min_value=3, max_value=7, step=1),
            activation="relu",
            kernel_initializer='he_normal', 
            kernel_regularizer=regularizers.l2(hp.Choice("kernel_regularizer_2", [0.0001, 0.0005, 0.001]))
        )
    )

    model.add(BatchNormalization())

    model.add(
        AveragePooling1D(
            pool_size=hp.Int("pool_size_2", [2, 3, 4])
        )
    )

    model.add(Dropout(hp.Float("dropout_2", min_value=0.2, max_value=0.5, step=0.1)))

    model.add(GlobalMaxPooling1D())

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

# Random search tuner to randomly find the best hyperparameters for the CNN model
tuner = RandomSearch(
    build_cnn_model,
    objective='val_accuracy',       # The tuner will try to find the hyperparameters that result in the highest validation accuracy.
    max_trials=10,                  # The maximum number of different hyperparameter combinations to try during the tuning process.
    executions_per_trial=3,         # The number of times to train the model with each hyperparameter combination. The average validation accuracy across these executions will be used to evaluate the hyperparameter combination.
    directory='cnn_tuning_results',         # The directory where the tuning results will be saved. This allows you to keep track of the different hyperparameter combinations that were tried and their corresponding validation accuracies.
    project_name='sentiment_analysis'       # The name of the project. This is used to create a subdirectory within the specified directory to store the tuning results for this specific project.
)

# Search for the best hyperparameters for the CNN model using the training and validation data
tuner.search(
    X_train_sequences,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val_sequences, y_val),
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

# Get the best hyperparameters found during the tuning process
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters found during the tuning process
print("Best Hyperparameters:")
print(best_hps.values)

# Build the best model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model on the training data, using the validation data for evaluation after each epoch. 
history = model.fit(
    X_train_sequences,      # The input data for training, which consists of the integer sequences representing the cleaned review texts in the training set.
    y_train,                # The target labels for training, which consists of the binary sentiment labels (0 or 1) corresponding to the cleaned review texts in the training set.
    epochs=10,              # The training process will run for a maximum of 10 epochs
    batch_size=64,          # with a batch size of 64 samples per gradient update. 
    validation_data=(X_val_sequences, y_val),    # The validation data to be used for evaluation after each epoch, which consists of the integer sequences representing the cleaned review texts in the validation set and their corresponding binary sentiment labels.
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]  
    # Early stopping is used to monitor the validation loss, and if it does not improve for 3 consecutive epochs, training will be stopped and the best weights will be restored.
)

# Create plots for training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Plots the training and validation accuracy and loss over the epochs to visualize the model's performance during training.
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

# Generate predictions on the test set
predictions = model.predict(X_test_sequences)
y_pred = (predictions >= 0.5).astype(int).flatten()   # If prediction >= 0.5, classify as 1 (positive sentiment), else classify as 0 (negative sentiment)

# Generate classification report based on the true labels (y_test) and the predicted labels (y_pred). The classification report includes precision, recall, f1-score, and support for each class (negative and positive sentiment).
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Generate confusion matrix graph to visualize the performance of the model in terms of true positives, true negatives, false positives, and false negatives.
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