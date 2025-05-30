import sys
import os
dirpath = os.path.dirname(__file__)
sys.path.append(dirpath)  # adds current dir to path
from typing import Annotated
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from backend import ChatRequest, CharacterInfo, record_audio, voice_clone, generate_combined_voice, query_deepseek, initiate_query_deepseek, convert_audio_to_wav, check_file_exists
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import ngrok


load_dotenv()  # defaults to .env in current dir

NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
APPLICATION_PORT = 8000

# ngrok free tier only allows one agent. So we tear down the tunnel on application termination
@asynccontextmanager
async def lifespan(app: FastAPI):
    # logger.info("Setting up Ngrok Tunnel")
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    ngrok.forward(
        addr=APPLICATION_PORT,
        domain="logically-sunny-boxer.ngrok-free.app",
    )
    yield
    # logger.info("Tearing Down Ngrok Tunnel")
    ngrok.disconnect()

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",  # <-- must match exactly!
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # no wildcards
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods: POST, GET, etc
    allow_headers=["*"],  # allow all headers
)

if not dirpath.__eq__(os.getcwd()):
    # Mount /static to serve files from ./static directory
    app.mount("/app/static", StaticFiles(directory="app/static"), name="static")
else:
    # Mount /static to serve files from ./static directory
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/upload-voice/")
async def upload_voice(file: UploadFile = File(...)):
    try:
        # Check file extension
        if file.content_type not in ["audio/wav", "audio/mpeg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only .wav and .mp3 are allowed.")

        # Read the uploaded file content directly to memory
        file_content = await file.read()
        input_format = file.filename.split(".")[-1].lower()

        # Convert the audio file to WAV and save it
        wav_file_path = convert_audio_to_wav(file_content, input_format)
        content = {"status": "success", "file": wav_file_path}
        response = JSONResponse(content=content)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-audio-path/")
async def check_audio_path():
    file_exists = check_file_exists('audio', 'voice_sample.wav')
    content = {"status": "success", "exists": file_exists}
    response = JSONResponse(content=content)
    return response

@app.post("/record-audio/")
async def record_audio_endpoint(duration: int = Form(10), filename: str = Form("voice_sample.wav")):
    """
    Records audio from the microphone.
    """
    try:
        saved_path = record_audio(filename=filename, duration=duration)

        content = {"status": "success", "file": saved_path}
        response = JSONResponse(content=content)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TextData(BaseModel):
    text: str

@app.post("/voice_clone/")
async def voice_clone_endpoint(data: TextData):
    # try:
        saved_path = voice_clone(data.text)
        # saved_path = generate_combined_voice(data.text)
        # saved_path = 'static/recordings/test.wav'

        content = {"status": "success", "file": saved_path}
        response = JSONResponse(content=content)
        return response
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(data: ChatRequest):
    # history_filepath = request.cookies.get('history_filepath')
    reply = query_deepseek(data.message, data.path)
    
    content = {"reply": reply}
    response = JSONResponse(content=content)
    return response

@app.post("/initiate/")
async def initiate_chat(character_info: CharacterInfo):
    message, filepath = initiate_query_deepseek(character_info)
    
    content = {"message": "User info saved successfully!", "file": filepath, "Reply": message}
    response = JSONResponse(content=content)
    # response.set_cookie(key="history_filepath", value=filepath, secure=True, samesite='none')
    return response

@app.get("/")
async def root():
    return {"message": "Welcome to the Voice Clone Preparation API (F5-TTS Ready)"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)