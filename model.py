import os
import pandas as pd
import string
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Suppress warnings
warnings.filterwarnings('ignore')

# Download stopwords
nltk.download('stopwords')

# Load your CSV dataset (adjust the path as needed)
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')  # Update with your file path

# Inspect the first few rows and column names
print(data.columns)  # Check if the column names are 'v1' and 'v2'

# Rename columns for easier access
data.columns = ['label', 'text', '_1', '_2', '_3']  # Rename 'v1' to 'label' and 'v2' to 'text', ignore other columns
data = data[['label', 'text']]  # Keep only relevant columns

# Convert labels to binary (1 for spam, 0 for ham)
data['spam'] = data['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Check class distribution
sns.countplot(x='spam', data=data)
plt.title('Spam and Non Spam Distribution')
plt.show()

# Balance the data by downsampling
spam_data = data[data.spam == 1]
non_spam_data = data[data.spam == 0].sample(len(spam_data), random_state=88)  # Downsampling ham
balanced_data = pd.concat([spam_data, non_spam_data], ignore_index=True)

# Preprocess text: remove punctuation, lowercase, and stopwords
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    stop_words = set(stopwords.words('english'))  # Set of stopwords
    return " ".join(word.lower() for word in text.split() if word.lower() not in stop_words)

balanced_data['text'] = balanced_data['text'].apply(preprocess_text)

# Split data into training, validation, and test sets
train_X, temp_X, train_Y, temp_Y = train_test_split(balanced_data['text'], balanced_data['spam'], test_size=0.2,
                                                    random_state=42)
val_X, test_X, val_Y, test_Y = train_test_split(temp_X, temp_Y, test_size=0.5, random_state=42)

# Tokenize and pad the text sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)
train_sequences = pad_sequences(tokenizer.texts_to_sequences(train_X), maxlen=100)
val_sequences = pad_sequences(tokenizer.texts_to_sequences(val_X), maxlen=100)
test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_X), maxlen=100)

# Set embedding dimensions
embedding_dim = 64

# Build the model with LSTM layers
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=100),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    BatchNormalization(),
    Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Train the model
history = model.fit(
    train_sequences, train_Y, validation_data=(val_sequences, val_Y),
    epochs=16, batch_size=32, callbacks=[early_stopping]
)

# Save the trained model
model.save('spam_detection_model.h5')
print("Model saved successfully.")

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Make predictions on the test set
test_predictions = (model.predict(test_sequences) > 0.5).astype('int32')  # Thresholding at 0.5 for binary classification

# Calculate confusion matrix
cm = confusion_matrix(test_Y, test_predictions)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

test_loss, test_accuracy = model.evaluate(test_sequences, test_Y, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

f1 = f1_score(test_Y, test_predictions)
precision = precision_score(test_Y, test_predictions)
recall = recall_score(test_Y, test_predictions)

print(f"Test F1 Score: {f1:.2f}")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
