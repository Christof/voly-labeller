#include "./recording_automation_controller.h"
#include "./recording_automation.h"

RecordingAutomationController::RecordingAutomationController(
      std::shared_ptr<RecordingAutomation> recordingAutomation)
  : recordingAutomation(recordingAutomation)
{
  connect(this, &RecordingAutomationController::startVideo, this,
          &RecordingAutomationController::startVideoInMainThread,
          Qt::QueuedConnection);
}

RecordingAutomationController::~RecordingAutomationController()
{
  disconnect(this, &RecordingAutomationController::startVideo, this,
             &RecordingAutomationController::startVideoInMainThread);
}

void RecordingAutomationController::takeScreenshotOfNextFrame()
{
  recordingAutomation->takeScreenshotOfNextFrame();
}

void RecordingAutomationController::startVideoInMainThread(QString positions)
{
  recordingAutomation->startVideo(positions.toStdString());
}
