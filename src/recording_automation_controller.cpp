#include "./recording_automation_controller.h"
#include <vector>
#include "./labelling_coordinator.h"
#include "./video_recorder.h"
#include "./utils/image_persister.h"

RecordingAutomationController::RecordingAutomationController(
      std::shared_ptr<LabellingCoordinator> labellingCoordinator,
      std::shared_ptr<VideoRecorder> videoRecorder)
  : labellingCoordinator(labellingCoordinator), videoRecorder(videoRecorder)
{
}

void RecordingAutomationController::initialize(Graphics::Gl* gl)
{
  this->gl = gl;
}

void RecordingAutomationController::resize(int width, int height)
{
  this->width = width;
  this->height = height;
}

void RecordingAutomationController::update()
{
  if (takeScreenshot)
  {
    std::vector<unsigned char> pixels(width * height * 4);
    gl->glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                     pixels.data());
    ImagePersister::flipAndSaveRGBA8I(pixels.data(), width, height,
                                      "screenshot.png");
    takeScreenshot = false;
  }
}

void RecordingAutomationController::takeScreenshotOfNextFrame()
{
  takeScreenshot = true;
}
