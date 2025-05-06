import sounddevice as sd
import scipy.io.wavfile as wav
import os
import requests
import json
from pydantic import BaseModel
from datetime import datetime
from F5TTS import TTS

AUDIO_SAVE_PATH = "recordings"
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

CHAT_SAVE_PATH = "chats"
os.makedirs(CHAT_SAVE_PATH, exist_ok=True)

TEMPLATE_PATH = "templates"
os.makedirs(CHAT_SAVE_PATH, exist_ok=True)

# DeepSeek API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-731921a5623e4c99a19176290b05f9a2"


class ChatRequest(BaseModel):
    message: str


class CharacterInfo(BaseModel):
    name: str
    relationship: str
    favorite_color: str


def record_audio(filename: str, duration: int = 10, samplerate: int = 24000):
    """Record audio from the microphone."""
    filepath = os.path.join(AUDIO_SAVE_PATH, filename)
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until finished
    wav.write(filepath, samplerate, recording)
    return filepath


def voice_clone(text: str, samplerate: int = 24000):
    ref_audio = "recordings/voice_sample.wav"
    ref_text = ""
    audio_output, _ = TTS(ref_audio, ref_text, text, remove_silence=True)
    filepath = os.path.join(AUDIO_SAVE_PATH, "test.wav")
    wav.write(filepath, samplerate, audio_output[1])
    return filepath


def fill_template(character_info: CharacterInfo) -> str:
    # Read template
    filename = 'character_info_template.txt'
    filepath = os.path.join(TEMPLATE_PATH, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        template_content = f.read()

    # Replace placeholders
    filled_content = template_content.replace("*NAME_PLACEHOLDER*", character_info.name)
    filled_content = filled_content.replace("*RELATIONSHIP_PLACEHOLDER*", character_info.relationship)
    filled_content = filled_content.replace("*COLOR_PLACEHOLDER*", character_info.favorite_color)

    return filled_content


def initiate_query_deepseek(character_info: CharacterInfo) -> str:
    filled_text = fill_template(character_info)

    dict_data = {
    "messages": [
        {
        "content": filled_text,
        "role": "system"
        },
        {
        "content": "请开始对话。",
        "role": "user"
        }
    ],
    "model": "deepseek-chat",
    "frequency_penalty": 0,
    "max_tokens": 2048,
    "presence_penalty": 0,
    "response_format": {
        "type": "text"
    },
    "stop": None,
    "stream": False,
    "stream_options": None,
    "temperature": 1,
    "top_p": 1,
    "tools": None,
    "tool_choice": "none",
    "logprobs": False,
    "top_logprobs": None
    }

    payload = json.dumps(dict_data)

    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }

    response = requests.request("POST", DEEPSEEK_API_URL, headers=headers, data=payload)
    dict_data['messages'].append(response.json()['choices'][0]['message'])

    # filename = f"{character_info.name}.json"
    filename = "character.json"
    filepath = os.path.join(CHAT_SAVE_PATH, filename)

    with open(filepath, 'w') as f:
        json.dump(dict_data, f)

    # response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content'], filepath
    else:
        return f"Error from DeepSeek: {response.text}"


def query_deepseek(message: str, history_filepath: str) -> str:

    with open(history_filepath) as f:
        dict_data = json.load(f)

    dict_data['messages'].append(
        {
            "content": message,
            "role": "user" 
        }
    )

    payload = json.dumps(dict_data)

    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }

    response = requests.request("POST", DEEPSEEK_API_URL, headers=headers, data=payload)

    dict_data['messages'].append(response.json()['choices'][0]['message'])

    with open(history_filepath, 'w') as f:
        json.dump(dict_data, f)

    # response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content']
    else:
        return f"Error from DeepSeek: {response.text}"