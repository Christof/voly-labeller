#include "./video_recorder.h"
#include <QTimer>
#include <QLoggingCategory>
#include <string>
#include "./graphics/gl.h"
#include "../external/VolyVideoModule/ffmpegrecorder.h"

QLoggingCategory videoRecorderChan("VideoRecorder");

VideoRecorder::VideoRecorder(double fps) : fps(fps)
{
}

VideoRecorder::~VideoRecorder()
{
  qCInfo(videoRecorderChan) << "Destructor"
                            << "isRecording:" << isRecording;
  if (isRecording)
    stopRecording();

  if (videoTimer.get())
    disconnect(videoTimer.get(), &QTimer::timeout, this,
               &VideoRecorder::updateVideoTimer);

  if (videoRecorder.get())
  {
    videoRecorder->stopRecording();
    qCInfo(videoRecorderChan)
        << "number of elapsed frames:" << videoRecorder->nbFramesElapsed();
    qCInfo(videoRecorderChan)
        << "number of stored frames:" << videoRecorder->nbFramesStored();
    qCInfo(videoRecorderChan) << "number of frames lost in capture:"
                              << videoRecorder->nbFramesLostInCapture();
    qCInfo(videoRecorderChan) << "number of frames lost in encoding:"
                              << videoRecorder->nbFramesLostInEncoding();
  }
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
    disconnect(videoTimer.get(), &QTimer::timeout, this,
               &VideoRecorder::updateVideoTimer);
  }

  pixelBuffer.resize(videoWidth * videoHeight * 3);
  const int channels = 1;
  videoRecorder = std::make_unique<FFMPEGRecorder>(videoWidth, videoHeight,
                                                   channels, filename, fps);

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

