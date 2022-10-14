FROM python:3.9.13

WORKDIR /app
COPY . . 

# install dependencies
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# RUN pip install tensorflow==2.9.2

# tell the port number container should expose
EXPOSE 8000

# run command
CMD ["python", "manage.py", "runserver"]