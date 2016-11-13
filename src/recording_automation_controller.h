#ifndef SRC_RECORDING_AUTOMATION_CONTROLLER_H_

#define SRC_RECORDING_AUTOMATION_CONTROLLER_H_

#include <QObject>
#include <memory>

class RecordingAutomation;

/**
 * \brief Wrapper around RecordingAutomation to provide access for the UI
 *
 */
class RecordingAutomationController : public QObject
{
  Q_OBJECT
 public:
  RecordingAutomationController(
      std::shared_ptr<RecordingAutomation> recordingAutomation);
  ~RecordingAutomationController();

 public slots:
  void takeScreenshotOfNextFrame();

signals:
  void startVideo(QString positions);

 private:
  std::shared_ptr<RecordingAutomation> recordingAutomation;
  void startVideoInMainThread(QString positions);
};

#endif  // SRC_RECORDING_AUTOMATION_CONTROLLER_H_
