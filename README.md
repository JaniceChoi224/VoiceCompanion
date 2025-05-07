# VoiceCompanion

Create a virtual environment using `Anaconda` and activate it

```
conda create -n voicecompanion python=3.10
conda activate voicecompanion
```

Install dependices

`pip install -r requirements.txt`

Run backend

`uvicorn main:app --reload --port 8000`

Run frontend on `http://localhost:3000`

`serve`

In miniconda3 device run

```
conda install conda-forge::portaudio
conda install conda-forge::ffmpeg
```
