import streamlit as st
import whisper
import threading
import time
import queue
import numpy as np

N_THREAD = 8

g_contexts = [None] * 4
g_mutex = threading.Lock()
g_worker = None

g_running = False

g_status = ""
g_status_forced = ""
g_transcribed = ""

g_pcmf32 = []

def stream_set_status(status):
    global g_status
    with g_mutex:
        g_status = status

def stream_main(index):
    stream_set_status("loading data ...")

    wparams = whisper.load_model("base")

    wparams.n_threads = min(N_THREAD, threading.active_count())
    wparams.offset_ms = 0
    wparams.translate = False
    wparams.no_context = True
    wparams.single_segment = True
    wparams.print_realtime = False
    wparams.print_progress = False
    wparams.print_timestamps = True
    wparams.print_special = False

    wparams.max_tokens = 32
    wparams.audio_ctx = 768  # partial encoder context for better performance

    # disable temperature fallback
    wparams.temperature_inc = -1.0

    wparams.language = "en"

    print(f"stream: using {wparams.n_threads} threads")

    pcmf32 = []

    # whisper context
    ctx = g_contexts[index]

    # 5 seconds interval
    window_samples = 5 

    while g_running:
        stream_set_status("waiting for audio ...")

        with g_mutex:
            if len(g_pcmf32) < 1024:
                continue

            pcmf32 = g_pcmf32[-min(len(g_pcmf32), window_samples):]
            g_pcmf32 = []

        t_start = time.time()

        stream_set_status("running whisper ...")

        ret = ctx.transcribe(pcmf32, **wparams)
        if ret != 0:
            print(f"whisper_full() failed: {ret}")
            break

        t_end = time.time()

        print(f"stream: whisper_full() returned {ret} in {t_end - t_start} seconds")

        text_heard = ""

        n_segments = len(ctx.segments)
        if n_segments > 0:
            segment = ctx.segments[-1]
            text_heard = segment.text

            print(f"transcribed: {text_heard}")

        with g_mutex:
            g_transcribed = text_heard

    if index < len(g_contexts):
        g_contexts[index] = None

def init(path_model):
    global g_running, g_worker

    for i in range(len(g_contexts)):
        if g_contexts[i] is None:
            g_contexts[i] = whisper.load_model(path_model)
            if g_contexts[i] is not None:
                g_running = True
                if g_worker is not None and g_worker.is_alive():
                    g_worker.join()
                g_worker = threading.Thread(target=stream_main, args=(i,))
                g_worker.start()

                return i + 1

    return 0

def free(index):
    global g_running
    if g_running:
        g_running = False

def set_audio(index, audio):
    index -= 1

    if index >= len(g_contexts) or g_contexts[index] is None:
        return -2

    with g_mutex:
        g_pcmf32.extend(audio)

    return 0

def get_transcribed():
    global g_transcribed
    with g_mutex:
        transcribed = g_transcribed
        g_transcribed = ""
    return transcribed

def get_status():
    global g_status, g_status_forced
    with g_mutex:
        status = g_status_forced if g_status_forced else g_status
    return status

def set_status(status):
    global g_status_forced
    with g_mutex:
        g_status_forced = status

def main():
    st.title("Whisper Audio Transcription")

    model_path = st.text_input("Enter model path", "base")
    start_button = st.button("Start Transcription")

    if start_button:
        init(model_path)

    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    if audio_file is not None:
        audio_bytes = audio_file.read()
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        set_audio(1, audio_np)

    status_text = st.empty()
    status_text.write(f"Status: {get_status()}")

    transcription_text = st.empty()
    transcription_text.write(f"Transcription: {get_transcribed()}")

    while True:
        status_text.write(f"Status: {get_status()}")
        transcription_text.write(f"Transcription: {get_transcribed()}")
        time.sleep(0.1)

if __name__ == "__main__":
    main()