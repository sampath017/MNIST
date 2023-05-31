FROM mcr.microsoft.com/devcontainers/python:3.9-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ARG USER=root

# Install prerequisites
RUN apt-get update && apt-get upgrade -y 
RUN apt-get install -y git 

# Install dependences
COPY requirements.txt ./
RUN pip install -r requirements.txt

