#include "./video_recorder.h"
#include <QTimer>
#include <QLoggingCategory>
#include "./graphics/gl.h"

QLoggingCategory videoRecorderChan("VideoRecorder");

VideoRecorder::~VideoRecorder()
{
  qCInfo(videoRecorderChan) << "Destructor";
  if (isRecording)
    stopRecording();

  disconnect(videoTimer.get(), &QTimer::timeout, this,
             &VideoRecorder::updateVideoTimer);
}

void VideoRecorder::initialize(Graphics::Gl *gl)
{
  this->gl = gl;
}

void VideoRecorder::createVideoRecorder(int xs, int ys, const char *filename,
                                        const double fps)
{
  qCInfo(videoRecorderChan) << "Create recorder" << filename;
  if (videoRecorder.get())
  {
    videoRecorder->stopRecording();
  }

  // mpeg supports even frame sizes only!
  videoWidth = (xs % 2 == 0) ? xs : xs - 1;
  videoHeight = (ys % 2 == 0) ? ys : ys - 1;
  pixelBuffer.resize(videoWidth * videoHeight * 3);
  videoRecorder = std::make_unique<FFMPEGRecorder>(videoWidth, videoHeight, 1,
                                                   true, filename, fps);

  videoRecorder->startRecording();
  videoTimer = std::make_unique<QTimer>(this);
  connect(videoTimer.get(), &QTimer::timeout, this,
          &VideoRecorder::updateVideoTimer);
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
  videoRecorder->queueFrame(pixelBuffer.data());
}

void VideoRecorder::captureVideoFrame()
{
  if (videoRecorder && isRecording)
  {
    gl->glReadPixels(0, 0, videoWidth, videoHeight, GL_RGB, GL_UNSIGNED_BYTE,
                     pixelBuffer.data());
  }
}

