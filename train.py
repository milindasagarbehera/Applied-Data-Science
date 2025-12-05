import pandas as pd
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np
import pickle

def load_training_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data['train'])
    return df

def preprocess_text(texts):
    """Tokenizes and pads the email texts."""
    
    # Initialize the tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token=OOV_TOKEN)
    
    # Fit the tokenizer on the training data texts
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences to ensure uniform length for the LSTM layer
    padded_sequences = pad_sequences(
        sequences, 
        maxlen=MAX_LEN, 
        padding=PADDING_TYPE, 
        truncating=TRUNC_TYPE
    )
    
    return padded_sequences, tokenizer

def build_lstm_model():
    """Defines and compiles the Long Short-Term Memory (LSTM) model."""
    # [Image of LSTM architecture]
    model = Sequential([
        # 1. Embedding Layer: Converts positive integer indices (words) into 
        # dense vectors of fixed size (EMBEDDING_DIM).
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        
        # 2. LSTM Layer: The core recurrent layer, excellent for sequences 
        # like text, capturing long-term dependencies.
        LSTM(LSTM_UNITS),
        
        # 3. Dropout: Prevents overfitting by randomly setting a fraction of 
        # input units to 0 at each update during training.
        Dropout(DROPOUT_RATE),
        
        # 4. Dense Layer: A standard fully connected layer.
        Dense(16, activation='relu'),
        
        # 5. Output Layer: Single unit with sigmoid activation for binary 
        # classification (output is probability between 0 and 1).
        Dense(1, activation='sigmoid') 
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', # Appropriate loss for binary classification
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Trains the model and evaluates performance."""
    print("\n--- Training Model ---")
    
    NUM_EPOCHS = 50
    
    # Start training
    history = model.fit(
        X_train, 
        y_train, 
        epochs=NUM_EPOCHS, 
        validation_data=(X_test, y_test),
        verbose=1 # Set to 1 to see progress
    )
    
    print("Training finished.")
    
    # Evaluate on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Evaluation Results ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}")
    
    return history, model

def predict_email(model, tokenizer, new_emails):
    """Processes new emails and makes predictions."""
    print("\n--- Making Predictions on New Emails ---")
    
    # Convert new emails to sequences
    new_sequences = tokenizer.texts_to_sequences(new_emails)
    
    # Pad the new sequences
    new_padded = pad_sequences(
        new_sequences, 
        maxlen=MAX_LEN, 
        padding=PADDING_TYPE, 
        truncating=TRUNC_TYPE
    )
    
    # Make predictions
    predictions = model.predict(new_padded)
    
    for email, prediction in zip(new_emails, predictions):
        # Prediction is a probability. Threshold is 0.5.
        label = "SPAM" if prediction[0] >= 0.5 else "HAM"
        print(f"Email: '{email}'")
        print(f"Prediction: {label} (Confidence: {prediction[0]:.4f})\n")

if __name__ == "__main__":
    training_data = load_training_data('data/train_data.json')
    texts = [j for j in training_data['Email_Text']]
    labels_ = [j for j in training_data['Label']]
    labels = np.array([1 if j == "spam" else 0 for j in labels_])
    print(training_data.head())

    ### Data Processing
    # Hyperparameters for text preprocessing
    MAX_WORDS = 1000  # Only consider the top 1000 words in the vocabulary
    MAX_LEN = 50      # Pad sequences to a max length of 50 words
    TRUNC_TYPE = 'post'
    PADDING_TYPE = 'post'
    OOV_TOKEN = "<oov>" # Out-of-vocabulary token
    padded_sequences, tokenizer = preprocess_text(texts)

    # saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. Split Data (using a very small split due to limited data)
    # In a real-world scenario, you'd use much more data.
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, 
        labels, 
        test_size=0.3, 
        random_state=42
    )

    # --- Model Definition and Training ---
    # Hyperparameters for the LSTM model
    EMBEDDING_DIM = 16
    LSTM_UNITS = 32
    DROPOUT_RATE = 0.2

    # Build and Train Model
    model = build_lstm_model()
    _, trained_model = train_model(model, X_train, y_train, X_test, y_test)
    trained_model.save('spam_lstm_model.h5')

    # Test with new, unseen data
    unseen_emails = [
        "Your account needs verification. Click here immediately to avoid suspension.",
        "Just wanted to follow up on our meeting yesterday. Could you send the notes?",
        "Win a free gift card now by following this secret link. Limited time only!",
    ]
    
    #predict_email(model, tokenizer, unseen_emails)