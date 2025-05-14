import sounddevice as sd
import scipy.io.wavfile as wav
import os
import re
import numpy as np
import requests
import json
from pydantic import BaseModel
from datetime import datetime
from pydub import AudioSegment
from io import BytesIO
from dotenv import load_dotenv
from F5TTS import TTS


load_dotenv()  # defaults to .env in current dir

dirpath = os.path.dirname(__file__)

if not dirpath.__eq__(os.getcwd()):
    dirpath = os.getcwd()
    AUDIO_SAVE_PATH = "app/static/recordings"
    CHAT_SAVE_PATH = "app/static/chats"
    TEMPLATE_PATH = "app/static/templates"
else:
    AUDIO_SAVE_PATH = "static/recordings"
    CHAT_SAVE_PATH = "static/chats"
    TEMPLATE_PATH = "static/templates"

dirpath += "/"
os.makedirs(dirpath + AUDIO_SAVE_PATH, exist_ok=True)
os.makedirs(dirpath + CHAT_SAVE_PATH, exist_ok=True)
os.makedirs(dirpath + TEMPLATE_PATH, exist_ok=True)

# DeepSeek API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


class ChatRequest(BaseModel):
    message: str
    path: str


class CharacterInfo(BaseModel):
    name: str
    gender: str
    age: str
    job: str
    relationship: str


def record_audio(filename: str, duration: int = 10, samplerate: int = 24000):
    """Record audio from the microphone."""
    filepath = os.path.join(AUDIO_SAVE_PATH, filename)
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until finished
    wav.write(dirpath + filepath, samplerate, recording)
    return filepath


def check_file_exists(category: str, filename: str) -> bool:
    """
    Checks if the specified file exists in the directory.

    Args:
        category (str): The category where the file belong.
        filename (str): The filename of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    if category == 'audio':
        file_path = os.path.join(AUDIO_SAVE_PATH, filename)
    elif category == 'templates':
        file_path = os.path.join(TEMPLATE_PATH, filename)
    elif category == 'chats':
        file_path = os.path.join(CHAT_SAVE_PATH, filename)
    return os.path.exists(dirpath + file_path)


def convert_audio_to_wav(file_content: bytes, input_format: str) -> str:
    """
    Converts an in-memory audio file (bytes) to WAV format and saves it to a directory.

    Args:
        file_content (bytes): The input audio file content in bytes.
        input_format (str): The format of the input file (e.g., "mp3", "wav").

    Returns:
        str: The path to the saved WAV file.
    """
    # Convert the file content to an audio segment
    audio = AudioSegment.from_file(BytesIO(file_content), format=input_format)

    # Define output filename
    wav_filename = "voice_sample.wav"

    wav_file_path = os.path.join(AUDIO_SAVE_PATH, wav_filename)

    # Export to .wav format
    audio.export(dirpath + wav_file_path, format="wav")

    return wav_file_path


def separate_by_punctuation(text):
    # Using regex to split text by any punctuation
    # separated_text = re.split(r'([.,!?;:()])', text)
    separated_text = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+|[.,!?;:()，。！？：；（）]', text)
    # Removing any empty strings from the list
    separated_text = [part.strip() for part in separated_text if part.strip()]
    return separated_text


def voice_clone(text: str, samplerate: int = 24000):
    ref_audio = AUDIO_SAVE_PATH + "/voice_sample.wav"
    ref_text = ""
    audio_output, _ = TTS(dirpath + ref_audio, ref_text, text, remove_silence=True)
    filepath = os.path.join(AUDIO_SAVE_PATH, "test.wav")
    wav.write(dirpath + filepath, samplerate, audio_output[1])
    return filepath


def generate_combined_voice(text: str, samplerate: int = 24000):
    """
    Generate a combined voice audio file using a voice cloning function.
    
    Args:
        text (str): The input text to convert to speech.
    
    Returns:
        AudioSegment: The combined voice audio with 0.1-second breaks at punctuation.
    """
    # Separate text by punctuation
    text_segments = separate_by_punctuation(text)

    # Create a combined audio segment
    combined_audio = np.array([], dtype=np.float32)
    
    ref_audio = AUDIO_SAVE_PATH + "/voice_sample.wav"
    ref_text = ""

    for segment in text_segments:
        print(segment)
        # if segment in ".,!?;:()":
        if re.match(r'[.,!?;:()，。！？：；（）]', segment):
            # Append 0.1 second of silence
            silence = np.zeros(int(samplerate * 0.1), dtype=np.float32)
            combined_audio = np.concatenate((combined_audio, silence))
        else:
            speech_audio, _ = TTS(dirpath + ref_audio, ref_text, segment, remove_silence=True)
            print(speech_audio[1].dtype)
            combined_audio = np.concatenate((combined_audio, speech_audio[1]))

    filepath = os.path.join(AUDIO_SAVE_PATH, "test.wav")
    # wav.write(dirpath + filepath, samplerate, combined_audio)
    # combined_audio.export(dirpath + filepath, format="wav")
    wav.write(dirpath + filepath, samplerate, combined_audio)
    return filepath


def fill_template(character_info: CharacterInfo) -> str:
    # Read template
    filename = 'character_info_template.txt'
    filepath = os.path.join(TEMPLATE_PATH, filename)
    with open(dirpath + filepath, "r", encoding="utf-8") as f:
        template_content = f.read()

    # Replace placeholders
    filled_content = template_content.replace("*NAME_PLACEHOLDER*", character_info.name)
    filled_content = filled_content.replace("*GENDER_PLACEHOLDER*", character_info.gender)
    filled_content = filled_content.replace("*AGE_PLACEHOLDER*", character_info.age)
    filled_content = filled_content.replace("*JOB_PLACEHOLDER*", character_info.job)
    filled_content = filled_content.replace("*RELATIONSHIP_PLACEHOLDER*", character_info.relationship)

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

    with open(dirpath + filepath, 'w') as f:
        json.dump(dict_data, f)

    # response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content'], filepath
    else:
        return f"Error from DeepSeek: {response.text}"


def query_deepseek(message: str, history_filepath: str) -> str:
    with open(dirpath + history_filepath) as f:
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

    with open(dirpath + history_filepath, 'w') as f:
        json.dump(dict_data, f)

    # response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content']
    else:
        return f"Error from DeepSeek: {response.text}"