FROM python:3.7-slim-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get install -y python3-dev && \
    apt-get install -y zstd && \
    pip3 install --no-cache shrynk && \
    apt-get remove -y gcc && apt-get -y autoremove
