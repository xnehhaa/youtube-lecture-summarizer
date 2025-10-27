import streamlit as st
from faster_whisper import WhisperModel
import yt_dlp
import tempfile
import os
from transformers import pipeline

# Load Whisper (cloud friendly model)
model = WhisperModel("small", device="cpu")

# Summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def download_audio_from_youtube(url):
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, "audio.mp3")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return audio_path

st.title("🎧 AI Lecture / YouTube Video Summarizer")

youtube_url = st.text_input("Paste YouTube link:")

if st.button("Summarize"):
    if youtube_url.strip() == "":
        st.error("Please enter a YouTube link.")
    else:
        with st.spinner("⬇ Downloading audio..."):
            audio_file = download_audio_from_youtube(youtube_url)

        with st.spinner("🎙 Transcribing using Whisper..."):
            segments, _ = model.transcribe(audio_file)
            transcript = " ".join([segment.text for segment in segments])

        st.subheader("📝 Transcript:")
        st.write(transcript)

        with st.spinner("🧠 Summarizing..."):
            summary = summarizer(transcript, max_length=200, min_length=60, do_sample=False)[0]['summary_text']

        st.subheader("🔍 Summary:")
        st.write(summary)


