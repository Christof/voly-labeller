#ifndef SRC_RECORDING_AUTOMATION_CONTROLLER_H_

#define SRC_RECORDING_AUTOMATION_CONTROLLER_H_

#include <QObject>
#include <memory>

class RecordingAutomation;

/**
 * \brief
 *
 *
 */
class RecordingAutomationController : public QObject
{
  Q_OBJECT
 public:
  RecordingAutomationController(
      std::shared_ptr<RecordingAutomation> recordingAutomation);

 public slots:
  void takeScreenshotOfNextFrame();

 private:
  std::shared_ptr<RecordingAutomation> recordingAutomation;
};

#endif  // SRC_RECORDING_AUTOMATION_CONTROLLER_H_
