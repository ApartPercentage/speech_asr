import streamlit as st
import pyaudio
import os
from pathlib import Path
import whisper
import numpy as np
import wave


transcript_path = 'transcripts/'
# Session state
if 'text' not in st.session_state:
	st.session_state['text'] = 'Listening...'
	st.session_state['run'] = False

# Audio parameters 
st.sidebar.header('Audio Parameters')

FRAMES_PER_BUFFER = int(st.sidebar.text_input('Frames per buffer', 3200))
RATE = int(st.sidebar.text_input('Rate', 16000))

def record_audio(FPB, Rate):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = Rate
    FRAMES_PER_BUFFER = FPB
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
        input_device_index=2
    )

    frames = []

    for i in range(0, int(RATE / FRAMES_PER_BUFFER * RECORD_SECONDS)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

# Start/stop audio transmission
def start_listening():
	st.session_state['run'] = True

def download_transcription():
	read_txt = open('transcription.txt', 'r')
	st.download_button(
		label="Download transcription",
		data=read_txt,
		file_name='transcription_output.txt',
		mime='text/plain')

def stop_listening():
	st.session_state['run'] = False


# Web user interface
st.title('🎙️ Real-Time Transcription App')

col1, col2 = st.columns(2)

col1.button('Start', on_click=start_listening)
col2.button('Stop', on_click=stop_listening)

model = whisper.load_model("base")
def model_transcribe(model, audio):
	result = model.transcribe(audio)["text"]
	return result

def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)

# Create a buffer to store audio data
#buffer = np.array([], dtype=np.int16)

#!!!THIS IS NOT WORKING!!!
while st.session_state['run']:
    # Read audio data from the stream
    #data = stream.read(FRAMES_PER_BUFFER)
    #audio_data = np.frombuffer(data, dtype=np.int16)
	record_audio(FRAMES_PER_BUFFER, RATE)

    # Append the new audio data to the buffer
    #buffer = np.concatenate((buffer, audio_data))

	audio = whisper.pad_or_trim(whisper.load_audio("output.wav"))
	transcription = model_transcribe(model, audio)

	output_transcript_file = "output.txt"

	save_transcript(transcription, output_transcript_file)
	output_file = open(os.path.join(transcript_path,output_transcript_file),"r")
    #output_file_data = output_file.read()
	
	# Check if the buffer has enough data for transcription
    #if len(buffer) >= 30 * 16000:  # 30 seconds of audio data at 16kHz sample rate
        #Transcribe the audio chunk
        #transcription = model.transcribe(buffer)
        #print(transcription["text"])

        # Clear the buffer
        #buffer = np.array([], dtype=np.int16)

    # Add a small delay to avoid consuming too many resources
    #time.sleep(0.01)

#if Path('transcription.txt').is_file():
#	st.markdown('### Download')
#	download_transcription()
#	os.remove('transcription.txt')

# References (Code modified and adapted from the following)
# 1. https://github.com/misraturp/Real-time-transcription-from-microphone
# 2. https://medium.com/towards-data-science/real-time-speech-recognition-python-assemblyai-13d35eeed226