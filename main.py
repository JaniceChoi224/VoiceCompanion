from fastapi import FastAPI, Response, Request, Form
from pydantic import BaseModel
from backend import ChatRequest, CharacterInfo, record_audio, voice_clone, query_deepseek, initiate_query_deepseek
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # allow all origins (frontend from any domain)
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods: POST, GET, etc
    allow_headers=["*"],  # allow all headers
)

@app.post("/record-audio/")
async def record_audio_endpoint(duration: int = Form(10), filename: str = Form("voice_sample.wav")):
    """
    Records audio from the microphone.
    """
    try:
        saved_path = record_audio(filename=filename, duration=duration)
        return {"status": "success", "file": saved_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class TextData(BaseModel):
    text: str

@app.post("/voice_clone/")
async def voice_clone_endpoint(data: TextData):
    # try:
        saved_path = voice_clone(data.text)
        return {"status": "success", "file": saved_path}
    # except Exception as e:
    #     return {"status": "error", "message": str(e)}

@app.post("/chat/")
async def chat(data: ChatRequest, request: Request):
    history_filepath = request.cookies.get('history_filepath')
    reply = query_deepseek(data.message, history_filepath)
    return {"reply": reply}

@app.post("/initiate/")
async def initiate_chat(character_info: CharacterInfo, response: Response):
    message, filepath = initiate_query_deepseek(character_info)
    response.set_cookie(key="history_filepath", value=filepath)
    return {"message": "User info saved successfully!", "file": filepath, "Reply": message}

@app.get("/")
async def root():
    return {"message": "Welcome to the Voice Clone Preparation API (F5-TTS Ready)"}
