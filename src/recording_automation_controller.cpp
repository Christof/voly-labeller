#include "./recording_automation_controller.h"
#include "./recording_automation.h"

RecordingAutomationController::RecordingAutomationController(
      std::shared_ptr<RecordingAutomation> recordingAutomation)
  : recordingAutomation(recordingAutomation)
{
}

void RecordingAutomationController::takeScreenshotOfNextFrame()
{
  recordingAutomation->takeScreenshotOfNextFrame();
}
