import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# --- Model Artifact Paths and Constants ---
# NOTE: These paths must match where spam_classifier.py saves the files.
MODEL_PATH = 'spam_lstm_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'
MAX_LEN = 50 # Must match the MAX_LEN used during training
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'

# --- Utility Functions for Prediction ---

@st.cache_resource
def load_artifacts():
    """
    Loads the model and tokenizer from disk. 
    Uses st.cache_resource to load them only once upon application startup.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        st.error(f"Required model files not found: {MODEL_PATH} and/or {TOKENIZER_PATH}.")
        st.info("Please run the 'spam_classifier.py' script first to generate the model and tokenizer files.")
        st.stop()
        
    try:
        # Load the Keras model
        # The 'custom_objects' is often needed when loading Keras models, 
        # especially those with custom layers, but usually fine without for standard ones.
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load the Tokenizer using pickle
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()


def preprocess_and_predict(email_text, model, tokenizer):
    """Tokenizes, pads, and predicts the class of a single email."""
    
    # 1. Convert text to sequence using the loaded tokenizer
    sequence = tokenizer.texts_to_sequences([email_text])
    
    # 2. Pad sequence to the exact length used during training
    padded_sequence = pad_sequences(
        sequence, 
        maxlen=MAX_LEN, 
        padding=PADDING_TYPE, 
        truncating=TRUNC_TYPE
    )
    
    # 3. Predict the probability
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    return prediction


# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="LSTM Spam Detector", layout="centered")

    st.title("📧 LSTM Email Spam Detector")
    st.markdown("""
        Enter the body of an email below to classify it as **HAM (safe)** or **SPAM (malicious)**.
    """)

    # Load artifacts (will only run once due to caching)
    model, tokenizer = load_artifacts()

    # Input text area
    email_input = st.text_area(
        "Enter Email Text Here:", 
        height=200, 
        placeholder="e.g., You've won a free iPhone! Click this secret link now."
    )

    if st.button("Classify Email", type="primary"):
        if email_input:
            with st.spinner('Analyzing email...'):
                confidence = preprocess_and_predict(email_input, model, tokenizer)
                
                # Determine classification based on 0.5 threshold
                if confidence >= 0.5:
                    label = "SPAM"
                    color = "red"
                    icon = "🚨"
                else:
                    label = "HAM"
                    color = "green"
                    icon = "✅"
                
                # Format output
                confidence_percent = f"{confidence * 100:.2f}%"

                st.subheader(f"Classification Result {icon}")
                
                # Display result using HTML for better styling
                st.markdown(f"""
                <div style="
                    border: 2px solid {color}; 
                    padding: 15px; 
                    border-radius: 10px; 
                    background-color: #f0f2f6;
                    font-size: 1.2em;
                    text-align: center;
                ">
                    **Predaiction:** <span style="color:{color}; font-weight: bold;">{label}</span><br>
                    **Confidence (Spam):** <span style="font-family: monospace;">{confidence_percent}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if label == "SPAM":
                    st.warning("This email is highly likely to be SPAM. Exercise extreme caution.")
                else:
                    st.success("This email appears to be safe (HAM).")

        else:
            st.warning("Please enter some text to classify.")

    st.markdown("---")
    st.caption("Model architecture: LSTM | Artifacts loaded: spam_lstm_model.h5 and tokenizer.pkl")

if __name__ == "__main__":
    main()