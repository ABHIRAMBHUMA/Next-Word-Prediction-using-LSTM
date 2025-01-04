import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model('lstmrnn.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_len:
        token_list = token_list[-(max_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Inject custom CSS
st.markdown(
    """
    <style>
    /* General body styling */
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f9fafb;
        margin: 0;
        padding: 0;
    }
    
    /* Main container styling */
    .main {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
        margin-top: 50px;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Title styling */
    h1 {
        text-align: center;
        color: #3a86ff;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border: 2px solid #d1d5db;
        border-radius: 8px;
        padding: 10px;
        font-size: 1rem;
        width: 100%;
        transition: border-color 0.3s ease-in-out;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3a86ff;
        box-shadow: 0 0 5px rgba(58, 134, 255, 0.4);
        outline: none;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3a86ff;
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: scale(1.05);
    }
    
    /* Alert messages styling */
    .stAlert {
        border-radius: 10px;
        padding: 20px;
        font-size: 1rem;
        font-weight: 500;
    }
    .stAlert > div {
        color: #333;
    }
    
    /* Success styling */
    .stAlert[data-baseweb="success"] {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
    }
    
    /* Error styling */
    .stAlert[data-baseweb="error"] {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
    }
    
    /* Warning styling */
    .stAlert[data-baseweb="warning"] {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#streamlit app
st.title('Next Word Prediction with LSTM')

input_text = st.text_input('Enter a sequence of words:', 'To be or not to')
if st.button('Predict Next Word'):
    if input_text.strip():
        max_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_len)
        if next_word:
            st.success(f'Predicted Next Word: **{next_word}**')
        else:
            st.error("Unable to predict the next word. Please try a different input.")
    else:
        st.warning("Please enter some text to predict the next word.")
