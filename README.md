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

or on GPU

`python -m http.server 3000`

In miniconda3 device run

```
conda install conda-forge::portaudio
conda install conda-forge::ffmpeg
```

On Ubuntu, install nginx as follow

```
sudo apt update
sudo apt install nginx
```

In the server, run `sudo vim ~/etc/nginx/sites-enabled/fastapi_nginx`
and edit the file as follow
```
server {
  listen 80;
  server_name 172.17.0.2;
  location / {
    proxy_pass http://127.0.0.1:8000;
  }
}
```
then run `sudo service nginx restart`
