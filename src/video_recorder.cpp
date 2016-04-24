#include "./video_recorder.h"
#include <QTimer>
#include "./graphics/gl.h"
#include "./VolyVideoModule/ffmpegrecorder.h"

VideoRecorder::~VideoRecorder()
{
  if (isRecording)
    stopRecording();
}

void VideoRecorder::initialize(Graphics::Gl *gl)
{
  this->gl = gl;
}

void VideoRecorder::createVideoRecorder(int xs, int ys, const char *filename,
                                        const double fps)
{
  if (videoRecorder)
  {
    videoRecorder->stopRecording();
    delete videoRecorder;
    delete[] pixelBuffer;
  }

  // mpeg supports even frame sizes only!
  videoWidth = (xs % 2 == 0) ? xs : xs - 1;
  videoHeight = (ys % 2 == 0) ? ys : ys - 1;
  pixelBuffer = new unsigned char[videoWidth * videoHeight * 3];
  videoRecorder =
      new FFMPEGRecorder(videoWidth, videoHeight, 1, true, filename, fps);

  videoRecorder->startRecording();
  videoTimer = new QTimer(this);
  connect(videoTimer, &QTimer::timeout, this, &VideoRecorder::updateVideoTimer);
}

void VideoRecorder::startRecording()
{
  isRecording = true;
  videoTimer->start(40);
}

void VideoRecorder::stopRecording()
{
  isRecording = false;
  videoTimer->stop();
}

void VideoRecorder::updateVideoTimer()
{
  videoRecorder->queueFrame(pixelBuffer);
}

void VideoRecorder::captureVideoFrame()
{
  if (videoRecorder && pixelBuffer && isRecording)
  {
    gl->glReadPixels(0, 0, videoWidth, videoHeight, GL_RGB, GL_UNSIGNED_BYTE,
                     pixelBuffer);
  }
}

