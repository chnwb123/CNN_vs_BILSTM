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

import matplotlib.pyplot as plt                 # To create visualizations and plots
import seaborn as sns                           # To create visualizations and plots, especially for statistical graphics

# <//////////////////////////////////////-- Packages Import --//////////////////////////////////////> 


# Read the cleaned Amazon reviews dataset from a CSV file into a pandas DataFrame
data = pd.read_csv(r'C:\APD3F2601SE\SEM_2\Courses\ODL\Assignment\Code\amazon_reviews_cleaned.csv')


# <//////////////////////////////////////-- Dataset Pre-processing --//////////////////////////////////////> 

# 1. Test, Train, and Validation Split
# 2. Tokenization
# 3. Encoding
# 4. Padding

# Split the dataset into features (X) and target (Y)
X = data["Cleaned Review Text"]       # Independent variable (features) - the cleaned review text
Y = data["Rating"]                    # Dependent variable (target) - the rating (0 or 1)

# Split the dataset into training and testing sets
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
# actual_vocabulary = tokenizer.get_vocabulary()
# actual_vocabulary_size = len(actual_vocabulary)

# Print the actual vocabulary size
# print("Actual vocabulary size:", actual_vocabulary_size)
actual_vocabulary_size = 17904

# Convert training, testing, and validation texts into integer sequences
X_train_sequences = tokenizer(X_train)
X_test_sequences = tokenizer(X_test)
X_val_sequences = tokenizer(X_val)

# Print the shapes of the integer sequences for training, testing, and validation datasets
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


# <//////////////////////////////////////-- Model Building --//////////////////////////////////////> 

# CNN
embedding_dim = 128             # The dimension of the dense embedding vectors that will be learned for each word in the vocabulary. Each word will be represented by a vector of this size in the embedding layer of the CNN model.

model = Sequential()            # Create a sequential model, which is a linear stack of layers. This is the simplest type of model in Keras, where you can add layers one after another.

model.add(Input(shape=(max_length,)))       # Add an input layer to the model, specifying the shape of the input data. The input shape is a tuple that defines the dimensions of the input data. In this case, the input shape is (max_length,), which means that each input sample will be a 1D array (vector) of length max_length (131). This layer does not have any trainable parameters; it simply defines the shape of the input data that will be fed into the model.

model.add(
    Embedding(
        input_dim=actual_vocabulary_size,
        output_dim=embedding_dim
    )
    # Add an embedding layer to the model, which will map each word in the input sequence to a dense vector of size embedding_dim.
)

model.add(
    Conv1D(
        filters=128,                # The number of output filters (or feature maps) in the convolutional layer. Each filter will learn to detect a specific feature or pattern in the input data.
        kernel_size=5,              # The size of the convolutional kernel (or filter) that will be applied to the input data. In this case, the kernel size is 5, which means that the convolutional layer will look at 5 consecutive words at a time when learning features from the input sequences.
        activation="relu",          # The activation function to be applied to the output of the convolutional layer. In this case, the ReLU (Rectified Linear Unit) activation function is used, which introduces non-linearity to the model and helps it learn complex patterns in the data.
        kernel_initializer='he_normal',     # The method used to initialize the weights of the convolutional layer. In this case, the He normal initializer is used, which draws samples from a truncated normal distribution centered on 0 with a standard deviation of sqrt(2 / fan_in), where fan_in is the number of input units in the weight tensor. This initializer is often used for layers with ReLU activation functions.
        kernel_regularizer=regularizers.l2(0.001),   # The regularization function to be applied to the weights of the convolutional layer. In this case, L2 regularization is used, which adds a penalty term to the loss function based on the sum of the squares of the weights. This helps prevent overfitting by discouraging large weights.
    )
)

model.add(BatchNormalization())     # Add a batch normalization layer to the model, which normalizes the activations of the previous layer at each batch. This helps stabilize and accelerate the training process by maintaining the mean activation close to 0 and the activation standard deviation close to 1.

model.add(
    AveragePooling1D(
        pool_size=2
    )
    # Add an average pooling layer to the model, which downsamples the input representation by taking the average over a fixed window (pool_size). In this case, the pool size is 2, which means that the layer will take the average of every 2 consecutive values in the input sequence, effectively reducing the length of the sequence by half.
)

model.add(Dropout(0.3))     # Add a dropout layer to the model, which randomly sets a fraction of the input units to 0 at each update during training, which helps prevent overfitting.

model.add(
    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu",
        kernel_initializer='he_normal', 
        kernel_regularizer=regularizers.l2(0.001), 
    )
)

model.add(BatchNormalization())

model.add(
    AveragePooling1D(
        pool_size=2
    )
)

model.add(Dropout(0.3))

model.add(GlobalMaxPooling1D())      # Add a global max pooling layer to the model, which will reduce the sequence representation to a single vector for each review.

# Output layer for binary classification (1 sentiment class)
model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

model.compile(
    loss="binary_crossentropy",     # The loss function to be used during training. In this case, binary cross-entropy is used, which is suitable for binary classification problems where the output is either 0 or 1.
    optimizer="adam",               # The optimization algorithm to be used during training. In this case, the Adam optimizer is used, which is an adaptive learning rate optimization algorithm that combines the benefits of both AdaGrad and RMSProp.
    metrics=["accuracy"]            # The metric to be monitored during training and evaluation. In this case, accuracy is used, which measures the proportion of correctly classified samples out of the total number of samples.
)

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

# <//////////////////////////////////////-- Model Building --//////////////////////////////////////> 