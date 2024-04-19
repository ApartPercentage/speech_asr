import streamlit as st
import streamlit_webrtc as webrtc
import whisper
import numpy as np
import time
from vosk import Model, KaldiRecognizer
import pyaudio

# --- Streamlit App Structure ---

st.title("Real-time Speech Recognition")
st.write("Transcribe Speech.")

# Configuration Parameters
vad_threshold = st.slider("Voice Activity Detection (VAD) Threshold", 0.0, 1.0, 0.5)

# Vosk Model Loading
model = Model('vosk-model-en-us-0.22/')
recognizer = KaldiRecognizer(model, 16000)

# Session state
if 'text not in st.session_state:
	st.session_state['text'] = 'Listening...'
	st.session_state['run'] = False
    
# Audio parameters 
st.sidebar.header('Audio Parameters')

FRAMES_PER_BUFFER = int(st.sidebar.text_input('Frames per buffer', 3200))
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = int(st.sidebar.text_input('Rate', 16000))
p = pyaudio.PyAudio()

# Open an audio stream with above parameter settings
stream = p.open(
   format=FORMAT,
   channels=CHANNELS,
   rate=RATE,
   input=True,
   frames_per_buffer=FRAMES_PER_BUFFER
)

# Start/stop audio transmission
def start_listening():
	st.session_state['run'] = True

def stop_listening():
	st.session_state['run'] = False

col1, col2 = st.columns(2)

col1.button('Start', on_click=start_listening)
col2.button('Stop', on_click=stop_listening)

while st.session_state['run']:
    try:
		stream.start_stream()
		data = stream.read(4096)
		if recognizer.AcceptWaveform(data):
			#Convert to transcript 
	except:

