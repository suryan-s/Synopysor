# import json
import yt_dlp
import os


def get_audio(v_name, url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': './temp/{}.%(ext)s'.format(v_name),
        'writethumbnail': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    # create yt-dlp object
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    os.rename(f'./temp/{v_name}.webp', f'./temp/temp.jpg')
    
get_audio("test", "https://www.youtube.com/watch?v=QH2-TGUlwu4")
    