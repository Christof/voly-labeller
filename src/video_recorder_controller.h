#ifndef VIDEO_RECORDER_CONTROLLER_H_

#define VIDEO_RECORDER_CONTROLLER_H_

#include <QObject>
#include <memory>

class VideoRecorder;

/**
 * \brief
 *
 *
 */
class VideoRecorderController : public QObject
{
  Q_OBJECT
 public:
  VideoRecorderController(std::shared_ptr<VideoRecorder> videoRecorder);

 public slots:
  void startNewVideo();

 signals:
  void toggleRecording();

 private:
  std::shared_ptr<VideoRecorder> videoRecorder;
 private slots:
  void toggleRecordingInMainThread();
};

#endif  // VIDEO_RECORDER_CONTROLLER_H_
