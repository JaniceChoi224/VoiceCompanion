# start from the official Python base image.
FROM python:3.10

# change working directory
WORKDIR /code

# add requirements file to image
COPY ./requirements.txt /code/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# install extra
RUN apt-get -y update \
 && apt-get -y upgrade \
 && apt-get install -y ffmpeg portaudio19-dev

# # append project and dta directories to PYTHONPATH
# RUN export PYTHONPATH="${PYTHONPATH}:/code/app"

# add python code
COPY ./app /code/app

# specify default commands
CMD ["fastapi", "run", "app/main.py", "--port", "80"]