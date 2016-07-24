FROM ubuntu:xenial

RUN apt-get update

RUN mkdir -p /voly-labeller/scripts
COPY ./scripts/* /voly-labeller/scripts/
WORKDIR /voly-labeller
