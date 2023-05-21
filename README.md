# Synopysor - YouTube Summarizer

Synopysor is a Python-based project that serves as a YouTube summarizer. The project utilizes various machine learning models to transcribe audio from YouTube videos into text and then generate concise summaries based on the transcribed content. The interface for Synopysor is built using Streamlit.

## How it Works

1. **YouTube Video Input**: The user provides a YouTube URL as input to Synopysor.

2. **Audio Extraction**: The YouTube video is downloaded and converted into an audio file.

3. **Whisper Model**: The audio file is processed using a whisper model, which transcribes the audio into text. This step enables the conversion of the spoken content in the video into a textual format.

4. **Text Parsing**: The transcribed text is properly parsed to remove any unnecessary information such as filler words, repetitions, or irrelevant content.

5. **Distil Bart NLP Model**: The parsed text is then passed through a Distil Bart NLP (Natural Language Processing) model, which generates a summary of the video content. The summary aims to capture the essential information and key points covered in the YouTube video.

6. **Streamlit Interface**: Synopysor provides a user-friendly interface using Streamlit, allowing users to input a YouTube URL, process the video, and view the generated summary.

## Future Improvements

- [ ] **Improved Models**: Incorporate more advanced and state-of-the-art NLP models to enhance the quality of the summaries generated by Synopysor. These models may provide better accuracy and more comprehensive summarization capabilities.

- [ ] **Faster Inference**: Optimize the project's inference time by exploring techniques such as model optimization, parallel processing, or any other possible ways. This enhancement will enable users to obtain results more quickly and efficiently.

- [ ] **Automatic Synchronization**: Implementing an automated system that periodically checks for new videos from subscribed channels and generates summaries would provide a convenient way for users to stay updated without manual input.

- [ ] **Keyword Extraction**: Implement a feature that extracts important keywords or key phrases from the video's content and includes them in the summary. This can help users quickly identify the main topics covered in the video.

- [ ] **Entity Recognition**: Integrate entity recognition capabilities to identify and highlight important entities mentioned in the video, such as people, locations, organizations, etc. This can provide users with additional context and insights.

- [ ]**Sentiment Analysis**: Analyze the sentiment of the video's content and include a sentiment score or a summary of the overall sentiment in the generated summary. This can help users understand the emotional tone of the video.

- [ ]**Topic Clustering**: Group related sentences or paragraphs in the summary based on common topics or themes. This can provide a structured and organized summary that is easier to comprehend.

- [ ]**Interactive Summary Visualization**: Create a visual representation of the summary, such as a word cloud or a bar chart, to highlight the most frequently mentioned words or topics in the video. This can provide users with a quick overview of the video's content.

- [ ] **Multiple Summaries**: Allow users to generate multiple summaries of different lengths for the same video. This gives users the flexibility to choose a summary that matches their requirements or time constraints.

- [ ] **Additional Features**: To further enhance the user experience and utility, such as:
  - Interactive visualization of the summarization process
  - Support for multiple languages
  - Integration with other popular video platforms

---

The project aims to provide users with summarized versions of YouTube videos, enabling them to quickly grasp the key information without watching the entire video. The future improvements section highlights potential enhancements to make the summarization process more accurate, faster, and feature-rich.