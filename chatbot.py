# Import libraries
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from groq import Groq
from gtts import gTTS
import os
import gradio as gr
from pydub import AudioSegment

# Initialize Whisper model
model = whisper.load_model("base")

GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)

# Function to transcribe audio using Whisper
def transcribe_audio(filepath):
    try:
        print(f"Transcribing audio from: {filepath}")
        result = model.transcribe(filepath)
        print(f"Transcription: {result['text']}")
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return "Error during transcription."

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
        print(f"LLM Response: {response}")
        return response
    except Exception as e:
        print(f"Error during LLM interaction: {str(e)}")
        return "Error during LLM interaction."

# Convert LLM response to speech using gTTS
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("response_audio.mp3")
        print("Audio saved as response_audio.mp3")
        return "response_audio.mp3"
    except Exception as e:
        print(f"Error during text-to-speech conversion: {str(e)}")
        return None

# Complete chatbot pipeline: Transcribe -> LLM -> TTS
def chatbot_pipeline(audio):
    try:
        # Convert audio to wav format if needed
        print(f"Received audio file: {audio}")
        if not audio.endswith('.wav'):
            audio_data = AudioSegment.from_file(audio)
            audio_data.export('input_audio.wav', format='wav')
            audio_path = 'input_audio.wav'
        else:
            audio_path = audio

        # Transcribe audio
        transcribed_text = transcribe_audio(audio_path)
        
        # Get LLM response
        response = get_llm_response(transcribed_text)
        
        # Convert response to speech
        response_audio = text_to_speech(response)
        
        return response_audio  # Send the audio file to the user

    except Exception as e:
        print(f"Error in chatbot pipeline: {str(e)}")
        return None

# Gradio interface for real-time voice-to-voice chatbot
gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.Audio(type="filepath"),  # Using filepath to handle audio
    outputs="audio",
    live=True
).launch()
  
   
