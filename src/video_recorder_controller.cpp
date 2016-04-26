#include "./video_recorder_controller.h"
#include <QDateTime>
#include "./video_recorder.h"

VideoRecorderController::VideoRecorderController(
    std::shared_ptr<VideoRecorder> videoRecorder)
  : videoRecorder(videoRecorder)
{
  connect(this, &VideoRecorderController::toggleRecording, this,
          &VideoRecorderController::toggleRecordingInMainThread,
          Qt::QueuedConnection);
  connect(this, &VideoRecorderController::startNewVideo, this,
          &VideoRecorderController::startNewVideoInMainThread,
          Qt::QueuedConnection);
}

VideoRecorderController::~VideoRecorderController()
{
  disconnect(this, &VideoRecorderController::toggleRecording, this,
             &VideoRecorderController::toggleRecordingInMainThread);
  disconnect(this, &VideoRecorderController::startNewVideo, this,
             &VideoRecorderController::startNewVideoInMainThread);
}

void VideoRecorderController::startNewVideoInMainThread()
{
  const QDateTime now = QDateTime::currentDateTime();
  const QString timestamp = now.toString(QLatin1String("yyyy-MM-dd-hhmmsszzz"));
  auto filename = QString::fromLatin1("video_%1.mpeg").arg(timestamp);

  videoRecorder->createNewVideo(filename.toStdString());
  videoRecorder->startRecording();

  emit recordingStateSwitched();
}

QString VideoRecorderController::getToggleText()
{
  if (videoRecorder->getIsRecording())
    return "Pause recording";

  return "Resume recording";
}

bool VideoRecorderController::getCanToggleRecording()
{
  return videoRecorder->getHasActiveRecording();
}

void VideoRecorderController::toggleRecordingInMainThread()
{
  if (videoRecorder->getIsRecording())
  {
    videoRecorder->stopRecording();
  }
  else
  {
    videoRecorder->startRecording();
  }

  emit recordingStateSwitched();
}

