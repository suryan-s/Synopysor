import whisper
from transformers import pipeline

summarizer = pipeline("summarization")

# Convert audio to text using OpenAI's whisper library
print("Load model")
model = whisper.load_model("small.en", download_root='Model', in_memory=True)
# model = whisper.load_model("large-v2")
result = model.transcribe("sample.mp3")
ARTICLE = result["text"]
ARTICLE = ARTICLE.replace('.', '.<eos>').replace('?', '?<eos>').replace('!', '!<eos>')# type: ignore

print("Split sentences")
sentences = ARTICLE.split('<eos>')
max_chunk = 500
current_chunk = 0 
chunks = []
for sentence in sentences:
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
    
    
print("Summarize chunks")
res = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
text = ' '.join([summ['summary_text'] for summ in res])# type: ignore
print(text)