import os
import streamlit as st
import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Define paths
audio_dataset_path = "C:/Users/Spyder/Desktop/Bird Sound/Audios/"
metadata_path = "C:/Users/Spyder/Desktop/Bird Sound/Metadata/Birds.xlsx"

# Load metadata
metadata = pd.read_excel(metadata_path, sheet_name='Sheet1')

# Load the pre-trained model
model = joblib.load("C:/Users/Spyder/Desktop/Bird Sound/saved_models/classifier.pkl")

# Define function to extract MFCC features
def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Streamlit app setup
st.set_page_config(
    page_title="Tweet-er",
    page_icon=":bird:",
    layout="wide",
)

# Add CSS to set background color and style heading
html_code = """
<div style="background-image: url("https://static.boredpanda.com/blog/wp-content/uploads/2016/05/Be-birds-572c7a1054a04__880.gif"); background-size: cover; background-repeat: no-repeat; background-attachment: fixed;">
    <center><h1 style="font-family: 'Arial', sans-serif; color: #00A6EB;">Tweet-er</h1>
    <p style="font-family: 'Times New Roman'; color: #ff0000;">Discover the mesmerizing world of aves. This is a Machine Learning Model that will help you identify your feathered friends using their chirps.</p>
    <p style="font-family: 'Times New Roman'; color: #ff0000;">These are some of the birds found around Bangalore.</p></center>
    <ul style="font-family: 'Arial'; color: #228B22; align: center;">
        <p style="font-family: 'Times New Roman', sans-serif; color: #ff0000;">
            <li>Asian Koel</li>
            <li>Ashy Prinia</li>
            <li>Red Junglefowl</li>
            <li>Rock Dove</li>
            <li>Brahminy Kite</li>
            <li>Indian Cuckoo</li>
            <li>Jungle Myna</li>
            <li>Pale-billed Flowerpecker</li>
            <li>White-browed Wagtail</li>
            <li>Spotted Owlet</li>
        </p>
    </ul>
</div>
"""





# Display HTML using st.markdown
st.markdown(html_code, unsafe_allow_html=True)


# Display the heading with improved styling
st.title("Bird Sound Classifier")

# Add a container with padding and background color for the file upload area
with st.container():
    st.markdown(
        '<div class="upload-area"><input type="file" accept=".wav" id="audioFile" style="display:none;" /><label for="audioFile">Upload Audio</label></div>',
        unsafe_allow_html=True,
    )

# Function to classify audio
def classify_audio(audio_file):
    try:
        # Extract features from the uploaded audio
        prediction_feature = features_extractor(audio_file)
        prediction_feature = prediction_feature.reshape(1, -1)

        # Use the pre-trained model to classify the audio
        predicted_class = model.predict(prediction_feature)[0]

        return predicted_class
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Check if an audio file is uploaded
uploaded_audio = st.file_uploader("Choose an audio file (WAV format)", type=["wav"])

if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/wav", start_time=0)
    classify_button = st.button("Classify Audio")

    if classify_button:
        predicted_class = classify_audio(uploaded_audio)
        if predicted_class is not None:
            st.success(f"Predicted Bird Species: {predicted_class}")
