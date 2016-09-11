#include "./recording_automation.h"
#include <vector>
#include "./labelling_coordinator.h"
#include "./video_recorder.h"
#include "./utils/image_persister.h"

RecordingAutomation::RecordingAutomation(
      std::shared_ptr<LabellingCoordinator> labellingCoordinator,
      std::shared_ptr<VideoRecorder> videoRecorder)
  : labellingCoordinator(labellingCoordinator), videoRecorder(videoRecorder)
{
}

void RecordingAutomation::initialize(Graphics::Gl* gl)
{
  this->gl = gl;
}

void RecordingAutomation::resize(int width, int height)
{
  this->width = width;
  this->height = height;
}

void RecordingAutomation::update()
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

void RecordingAutomation::takeScreenshotOfNextFrame()
{
  takeScreenshot = true;
}
