# FROM mcr.microsoft.com/devcontainers/python:3.11
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \ 
    && apt-get install -y git curl

# # poetry installation
# ENV PATH="$HOME/.local/bin:$PATH"

# RUN echo "done 0" \
#     && curl -sSL https://install.python-poetry.org | python3 - \
#     && export PATH="$HOME/.local/bin:$PATH" \
#     && poetry config virtualenvs.in-project true
