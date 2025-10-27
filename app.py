import streamlit as st
import whisper
import yt_dlp
import tempfile
import os
from transformers import pipeline

# Load Whisper model
model = whisper.load_model("small")

# Summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def download_audio_from_youtube(url):
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, "youtube_audio.mp3")

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
        with st.spinner("⏳ Downloading audio..."):
            audio_file = download_audio_from_youtube(youtube_url)

        with st.spinner("🎙️ Transcribing using Whisper..."):
            result = model.transcribe(audio_file)
            transcript = result["text"]

        st.subheader("📝 Transcript:")
        st.write(transcript)

        with st.spinner("🧠 Generating summary..."):
            summary = summarizer(transcript, max_length=200, min_length=60, do_sample=False)[0]['summary_text']

        st.subheader("🔍 Summary:")
        st.write(summary)


