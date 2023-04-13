import os

import streamlit as st
import validators
import whisper
from torch import cuda, device
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from yt_dlp import YoutubeDL

devices = device("cuda:0" if cuda.is_available() else "cpu")

if os.path.exists('Model/distilbart-xsum-12-1'):
    pass
else:
    print('Downloading Summarizing Model')
    model_name = "sshleifer/distilbart-xsum-12-1"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained("Model/distilbart-xsum-12-1")
    tokenizer.save_pretrained("Model/distilbart-xsum-12-1")

# @st.cache_resource(experimental_allow_widgets=True,show_spinner=False)
def load_model():
    print("Loading Model")
    summarizer_ = pipeline("summarization",model="Model/distilbart-xsum-12-1", device=devices)
    if os.path.exists(os.path.join('Model','small.en.pt')):
        print("Model Exists")
        model_ = whisper.load_model(os.path.join('Model','small.en.pt'),in_memory=True, device = devices)
        return model_, summarizer_
    else:
        print("Model Doesn't Exist")
        model_ = whisper.load_model("small.en", download_root='Model', in_memory=True, device = devices)
        return model_, summarizer_


def whisp_audio(location):
    model, summarizer__ = load_model()
    
    # Convert audio to text using OpenAI's whisper library
    result = model.transcribe(location)
    ARTICLE = result["text"]
    ARTICLE = ARTICLE.replace('.', '.<eos>').replace('?', '?<eos>').replace('!', '!<eos>')# type: ignore
    sentences_ = ARTICLE.split('<eos>')
    return sentences_, summarizer__

def make_chunks(sentences_):
    max_chunk = 500
    current_chunk = 0
    chunks = []
    for sentence in sentences_:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            print(current_chunk)
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    return chunks

 
def summanrize_chunks(chunks, max_summ_len,summarizer):
    try:
        res = summarizer(chunks, max_length=int(max_summ_len), min_length=30, do_sample=False)
        text = ' '.join([summ['summary_text'] for summ in res])# type: ignore
        text = text.replace(' . ', '.')
        text = text.replace('.', '.\n')
        st.success("Summarization complete")
        return text
    except Exception:
        st.error("Error occured while summarizing")
        return None

@st.cache_resource(experimental_allow_widgets=True,show_spinner=False)
def transform_to_summarry(audio_location, max_len, url):
    print(url)
    with st.spinner("Summarizing..."):
        sentences, summarizer___ = whisp_audio(audio_location)
        blocks = make_chunks(sentences)
        return summanrize_chunks(blocks,max_len, summarizer___)

# @st.cache_resource(show_spinner=False)
def get_audio(url):
    v_name = 'temp'
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
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        info_dict = ydl.extract_info(url, download=False)
        v_name = info_dict['title'] #type: ignore
        
    # os.rename(f'temp/{v_name}.webp', f'temp/temp.jpg')
    if os.path.exists(os.path.join('temp','temp.jpg')):
        os.remove(os.path.join('temp','temp.jpg'))
    os.rename(os.path.join('temp','temp.webp'),os.path.join('temp','temp.jpg'))
    return v_name


@st.cache_resource(experimental_allow_widgets=True,show_spinner=False)
def load_yt_details(y_url):
    with st.spinner("Loading data..."):
        try:
            res = get_audio(y_url)
            return res
        except Exception:
            return False


def show_image(image_path,image_title):
    st.image(image_path, caption=image_title, use_column_width=True)



# Set title and page configuration
st.set_page_config(page_title='Synopysor',initial_sidebar_state="expanded",page_icon=":book:")
st.title('Synopysor')

# Set default values
image_path = os.path.join('temp','temp.jpg')
max_len = 120

# Input field for YouTube URL
ytb_url = st.text_input('Enter YouTube URL')

# Validate URL
if ytb_url and not validators.url(ytb_url): #type: ignore
    st.error('Invalid YouTube URL')

# Show image and get max_len input
if validators.url(ytb_url): #type: ignore
    image_title = load_yt_details(ytb_url)
    if image_title!=False:
        st.success('Retrieved data')
        # Display image 
        show_image(image_path,image_title)
        # Get user input for max_len
        max_len = st.number_input('Enter maximum number of words for summary', value=max_len, help="Maximum number of words in summary. The ideal number is 120 which is the default value. \nIf you want to summarize a long article, increase this number. \nIf you want to summarize a short article, decrease this number.")
        
        if int(max_len) > 10: # type: ignore 
            # Button to start summarization
            if st.button('Start Synopysor'):
                # Perform summarization here
                conv_res = transform_to_summarry(os.path.join('temp','temp.mp3'),max_len, url=ytb_url)
                if conv_res is None:
                    pass
                else:
                    st.code(conv_res, language='markdown')
                    # st.text(conv_res)
            else:
                st.stop()
    else:
        st.error('Failed to load data')
    # st.cache_data.clear()
        
    
    
    