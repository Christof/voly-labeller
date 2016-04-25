#include "./video_recorder.h"
#include <QTimer>
#include <QLoggingCategory>
#include "./graphics/gl.h"

QLoggingCategory videoRecorderChan("VideoRecorder");

VideoRecorder::VideoRecorder(double fps) : fps(fps)
{
}

VideoRecorder::~VideoRecorder()
{
  qCInfo(videoRecorderChan) << "Destructor";
  if (isRecording)
    stopRecording();

  if (videoTimer.get())
    disconnect(videoTimer.get(), &QTimer::timeout, this,
               &VideoRecorder::updateVideoTimer);
}

void VideoRecorder::initialize(Graphics::Gl *gl)
{
  this->gl = gl;
}

void VideoRecorder::resize(int width, int height)
{
  stopRecording();

  // mpeg supports even frame sizes only!
  videoWidth = (width % 2 == 0) ? width : width - 1;
  videoHeight = (height % 2 == 0) ? height : height - 1;
}

void VideoRecorder::createNewVideo(std::string filename)
{
  qCInfo(videoRecorderChan) << "Create recorder" << filename.c_str();
  if (videoRecorder.get())
  {
    videoRecorder->stopRecording();
  }

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
  if (!isRecording)
    return;

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

bool VideoRecorder::getIsRecording()
{
  return isRecording;
}

bool VideoRecorder::getHasActiveRecording()
{
  return videoRecorder.get() != nullptr;
}

