import gradio as gr
import openai
import os
import requests

from pathlib import Path

# openai.api_key = os.environ.get('OPEN_API_KEY')

OPEN_API_KEY = os.environ.get('OPEN_API_KEY', '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')
openai.api_key = OPEN_API_KEY


def transcribe(audio):
    myfile = Path(audio)
    myfile = myfile.rename(myfile.with_suffix('.wav'))

    audio_file = open(myfile, "rb")
 
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    # Send user input to server
    response = requests.post("http://localhost:5300/recommend", json={'text': transcript['text'], 'history': []})

    return response.json()['output']


demo = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text")

demo.launch()
#demo.launch(share=True)
