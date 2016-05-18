/*
VolyRenderer.

Authors: Alexander Bornik, Martin Urschler
Mail: medlib@lists.icg.tugraz.at, bornik@icg.tugraz.at, urschler@icg.tugraz.at

Copyright (C) 2010 Institute for Computer Graphics and Vision,
Graz University of Technology
&
Ludwig Boltzmann Institute for Clinical Forensic Imaging
Graz
*/


#include "ffmpegrecorderworker.h"
#include "ffmpegrecorder.h"
#include "ffmpeg_encoder.hpp"
#include "framepool.h"
#include <cstdio>

#include <QDebug>

FFMPEGRecorderWorker::FFMPEGRecorderWorker(FFMPEGRecorder *recorder, int channel, QObject *parent) :
  QThread(parent)
{
  m_recorder = recorder;
  m_channel = channel;

  start();

}
FFMPEGRecorderWorker::~FFMPEGRecorderWorker()
{
  qDebug() << "FFMPEGRecorderWorker::~FFMPEGRecorderWorker(): waiting for worker";

  wait();

  qDebug() << "FFMPEGRecorderWorker::~FFMPEGRecorderWorker(): worker finished";
}

void FFMPEGRecorderWorker::run()
{
  qDebug() << "Starting Video encoding(" << m_channel <<")";

  unsigned char *frame[4];

  bool need_exit = false;

  for(;;) {
    frame[m_channel]=(unsigned char *)m_recorder->fp[m_channel]->getFrame();
    if(!frame[m_channel]) {
      need_exit =true;
    }

    if (need_exit) break;

    m_recorder->mpeg_file[m_channel]->addFrame(frame[m_channel]);

    m_recorder->fp[m_channel]->removeFrame();
    m_recorder->nb_frames_stored++;
  }
}
