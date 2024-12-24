FROM anibali/pytorch:2.0.0-cuda11.8
USER root



RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install   git wget nano  -y

RUN pip install tqdm


RUN pip install gdown
RUN pip install git+https://github.com/qubvel/segmentation_models.pytorch.git@v0.2.1
RUN pip install -U albumentations[imgaug]
RUN pip install notebook==6
RUN pip install nbconvert


