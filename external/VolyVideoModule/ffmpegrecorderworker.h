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


#ifndef FFMPEGRECORDERWORKER_H
#define FFMPEGRECORDERWORKER_H

#include <QThread>

class FFMPEGRecorder;

class FFMPEGRecorderWorker : public QThread
{
  Q_OBJECT
public:
  FFMPEGRecorderWorker(FFMPEGRecorder *recorder, int channel, QObject *parent = 0);
  virtual ~FFMPEGRecorderWorker();

signals:

public slots:

protected:
  virtual void run();
  int m_channel;
  FFMPEGRecorder *m_recorder;
};

#endif // FFMPEGRECORDERWORKER_H
