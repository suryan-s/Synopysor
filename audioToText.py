import os

import whisper
from transformers import pipeline


def load_model():
    print("Loading Model")
    summarizer_ = pipeline("summarization")
    if os.path.exists(os.path.join('Model','small.en.pt')):
        print("Model Exists")
        model_ = whisper.load_model(os.path.join('Model','small.en.pt'),in_memory=True)
        return model_, summarizer_
    else:
        print("Model Doesn't Exist")
        model_ = whisper.load_model("small.en", download_root='Model', in_memory=True)
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
        text = text.replace('.', '.\n').replace(' .', '.')
        return text
    except Exception:
        return None


def transform_to_summarry(audio_location, max_len):
    sentences, summarizer___ = whisp_audio(audio_location)
    blocks = make_chunks(sentences)
    return summanrize_chunks(blocks,max_len, summarizer___)


# print(transform_to_summarry('temp/temp.mp3',120))
