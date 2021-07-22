FROM ubuntu:latest

RUN apt update; DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y cmake libopencv-dev g++
COPY . /app-data
RUN cd /app-data/src; mkdir build; cd build; cmake ../; make

WORKDIR /app-data/src/cv/task1
ENTRYPOINT ["/app-data/src/build/cv/task1/cvtask1"]
