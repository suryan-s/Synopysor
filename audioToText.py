import whisper

# Download audio from YouTube video
# ydl_opts = {
#     'format': 'bestaudio/best',
#     'outtmpl': '%(id)s.%(ext)s',
# }
# with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#     info_dict = ydl.extract_info("https://www.youtube.com/watch?v=nBZT7lF5dag", download=False)
#     audio_url = info_dict['formats'][0]['url']
#     audio_bytes = io.BytesIO(ydl.extract_info("https://www.youtube.com/watch?v=nBZT7lF5dag", download=True)['formats'][0]['filesize'])
#     with sf.SoundFile(audio_bytes) as f:
#         audio_data = f.read()


# Convert audio to text using OpenAI's whisper library
model = whisper.load_model("small.en", download_root='Model', in_memory=True)
# model = whisper.load_model("large-v2")
result = model.transcribe("sample.mp3")
print(result["text"])