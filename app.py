# Import libraries
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from groq import Groq
from gtts import gTTS
import os
from pydub import AudioSegment
import streamlit as st
import tempfile

# Initialize Whisper model
model = whisper.load_model("base")

GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)

# Function to transcribe audio using Whisper
def transcribe_audio(filepath):
    try:
        result = model.transcribe(filepath)
        return result['text']
    except Exception as e:
        return f"Error during transcription: {str(e)}"

# Get LLM response from Groq API
def get_llm_response(transcribed_text):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": transcribed_text,
                }
            ],
            model="llama3-8b-8192",
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        return f"Error during LLM interaction: {str(e)}"

# Convert LLM response to speech using gTTS
def text_to_speech(text):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts = gTTS(text=text, lang='en')
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        return f"Error during text-to-speech conversion: {str(e)}"

# Complete chatbot pipeline: Transcribe -> LLM -> TTS
def chatbot_pipeline(audio):
    try:
        # Convert audio to wav format if needed
        if audio is not None:
            if not audio.name.endswith('.wav'):
                audio_data = AudioSegment.from_file(audio)
                wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                audio_data.export(wav_file.name, format='wav')
                audio_path = wav_file.name
            else:
                audio_path = audio.name

            # Transcribe audio
            transcribed_text = transcribe_audio(audio_path)
            
            # Get LLM response
            response = get_llm_response(transcribed_text)
            
            # Convert response to speech
            response_audio = text_to_speech(response)
            
            return response_audio  # Send the audio file to the user
    except Exception as e:
        return f"Error in chatbot pipeline: {str(e)}"

# Streamlit UI
st.title("Voice-to-Voice Chatbot")
st.write("Upload an audio file to interact with the chatbot.")

audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if st.button("Submit"):
    if audio_file is not None:
        response_audio_file = chatbot_pipeline(audio_file)
        
        if isinstance(response_audio_file, str) and response_audio_file.endswith('.mp3'):
            st.audio(response_audio_file, format='audio/mp3')
        else:
            st.error(response_audio_file)  # Display error message
    else:
        st.error("Please upload an audio file.")
