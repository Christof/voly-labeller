#include "./recording_automation.h"
#include <vector>
#include <string>
#include "./labelling_coordinator.h"
#include "./video_recorder.h"
#include "./nodes.h"
#include "./camera_node.h"
#include "./camera.h"
#include "./utils/image_persister.h"

QLoggingCategory recordingAutomationChan("RecordingAutomation");

RecordingAutomation::RecordingAutomation(
    std::shared_ptr<LabellingCoordinator> labellingCoordinator,
    std::shared_ptr<Nodes> nodes, std::shared_ptr<VideoRecorder> videoRecorder)
  : labellingCoordinator(labellingCoordinator), nodes(nodes),
    videoRecorder(videoRecorder)
{
}

void RecordingAutomation::initialize(Graphics::Gl *gl)
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
  if (!takeScreenshot)
    return;

  if (shouldMoveToPosition)
  {
    moveToCameraPosition(cameraPositionName);
    shouldMoveToPosition = false;
    return;
  }

  if (labellingCoordinator->haveLabelPositionsChanged())
  {
    unchangedCount = 0;
  }
  else
  {
    unchangedCount++;
  }

  if (unchangedCount > 10)
  {
    std::vector<unsigned char> pixels(width * height * 4);
    gl->glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                     pixels.data());
    std::string filename = "screenshot_" + nodes->getSceneName() + "_" +
                           cameraPositionName + ".png";
    ImagePersister::flipAndSaveRGBA8I(pixels.data(), width, height, filename);
    takeScreenshot = false;
    qCWarning(recordingAutomationChan) << "Took screenshot:"
                                       << filename.c_str();

    if (exitAfterScreenshot)
      exit(0);
  }
}

void RecordingAutomation::takeScreenshotOfNextFrame()
{
  takeScreenshot = true;
}

void RecordingAutomation::takeScreenshotOf(std::string cameraPositionName)
{
  this->cameraPositionName = cameraPositionName;
  shouldMoveToPosition = true;
  takeScreenshot = true;
}

void RecordingAutomation::takeScreenshotOfPositionAndExit(
    std::string cameraPositionName)
{
  exitAfterScreenshot = true;
  takeScreenshotOf(cameraPositionName);
}

void RecordingAutomation::moveToCameraPosition(std::string name)
{
  auto cameraNode = nodes->getCameraNode();
  auto cameraPositions = cameraNode->cameraPositions;

  auto cameraPosition =
      std::find_if(cameraPositions.begin(), cameraPositions.end(),
                   [name](const CameraPosition &cameraPosition) {
                     return cameraPosition.name == name;
                   });

  if (cameraPosition == cameraPositions.end())
  {
    qCWarning(recordingAutomationChan)
        << "Position with name " << cameraPositionName.c_str() << " not found!";
    return;
  }

  auto target = cameraPosition->viewMatrix;
  cameraNode->getCamera()->startAnimation(target, 1e-9f);
}

