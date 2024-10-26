import whisper
from groq import Groq
from gtts import gTTS
import gradio as gr
import os
import tempfile

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# Set up Groq client for LLM interaction



os.environ["GROQ_API_KEY"] = "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"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def process_speech_to_speech(audio):
    # Step 1: Transcribe the audio using Whisper
    transcript = whisper_model.transcribe(audio)["text"]
    
    # Step 2: Send transcription to the LLM via Groq
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": transcript}],
        model="llama3-8b-8192"
    )
    response_text = chat_completion.choices[0].message.content

    # Step 3: Convert LLM response to speech using gTTS
    tts = gTTS(text=response_text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_output = fp.name  # Path to the generated speech file

    return response_text, audio_output

# Gradio interface to deploy the application
iface = gr.Interface(
    fn=process_speech_to_speech,
    inputs=gr.Audio(type="filepath", label="Record your audio"),
    outputs=[gr.Textbox(label="LLM Response"), gr.Audio(type="filepath", label="Spoken Response")],
    live=True
)


iface.launch()
