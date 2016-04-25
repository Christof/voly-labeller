#include "./video_recorder_controller.h"
#include <QDateTime>
#include "./video_recorder.h"

VideoRecorderController::VideoRecorderController(
    std::shared_ptr<VideoRecorder> videoRecorder)
  : videoRecorder(videoRecorder)
{
  connect(this, SIGNAL(toggleRecording()), this,
          SLOT(toggleRecordingInMainThread()), Qt::QueuedConnection);
}

void VideoRecorderController::startNewVideo()
{
  const QDateTime now = QDateTime::currentDateTime();
  const QString timestamp = now.toString(QLatin1String("yyyy-MM-dd-hhmmsszzz"));
  auto filename = QString::fromLatin1("video_%1.mpeg").arg(timestamp);

  videoRecorder->createNewVideo(filename.toStdString());
  videoRecorder->startRecording();
}

QString VideoRecorderController::getToggleText()
{
  if (videoRecorder->getIsRecording())
    return "Stop recording";

  return "Start recording";
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

