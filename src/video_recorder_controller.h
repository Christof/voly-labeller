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
  Q_PROPERTY(QString toggleRecordingText READ getToggleText NOTIFY
                 recordingStateSwitched)
  Q_PROPERTY(bool canToggleRecording READ getCanToggleRecording NOTIFY
                 recordingStateSwitched)
 public:
  VideoRecorderController(std::shared_ptr<VideoRecorder> videoRecorder);

 public slots:
  QString getToggleText();
  bool getCanToggleRecording();

signals:
  void toggleRecording();
  void startNewVideo();
  void recordingStateSwitched();

 private:
  std::shared_ptr<VideoRecorder> videoRecorder;
 private slots:
  void toggleRecordingInMainThread();
  void startNewVideoInMainThread();
};

#endif  // VIDEO_RECORDER_CONTROLLER_H_
