FROM ubuntu:xenial

RUN apt-get update
RUN apt-get -y install sudo cmake wget

RUN apt-get -y install \
  build-essential \
  libgl1-mesa-dev \
  libfontconfig1-dev \
  libfreetype6-dev \
  libx11-dev \
  libxext-dev \
  libxfixes-dev \
  libxi-dev \
  libxrender-dev \
  libxcb1-dev \
  libx11-xcb-dev \
  libxcb-glx0-dev \
  libxcb-keysyms1-dev \
  libxcb-image0-dev \
  libxcb-shm0-dev \
  libxcb-icccm4-dev \
  libxcb-sync0-dev \
  libxcb-xfixes0-dev \
  libxcb-shape0-dev \
  libxcb-randr0-dev \
  libxcb-render-util0-dev \
  libxcb-xinerama0 \
  libxcb-xinerama0-dev \
  python

WORKDIR /tmp
RUN wget http://download.qt.io/official_releases/qt/5.7/5.7.0/single/qt-everywhere-opensource-src-5.7.0.tar.gz
RUN gunzip qt-everywhere-opensource-src-5.7.0.tar.gz && \
  tar xvf qt-everywhere-opensource-src-5.7.0.tar

WORKDIR /tmp/qt-everywhere-opensource-src-5.7.0
RUN pwd
RUN ./configure -opensource -confirm-license -skip qtconnectivity  -prefix /usr/local
RUN make -j2 && make install

RUN mkdir -p /voly-labeller/scripts
COPY ./scripts/* /voly-labeller/scripts/
WORKDIR /voly-labeller

RUN /voly-labeller/scripts/install_dependencies.sh
# Move to top
RUN apt-get install unzip 
RUN apt-get install libinsighttoolkit4-dev -y

CMD /bin/bash
