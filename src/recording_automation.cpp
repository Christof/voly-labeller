#include "./recording_automation.h"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>
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

std::string getTime()
{
  std::stringstream date;

  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  date << std::put_time(&tm, "%Y-%m-%d_%H:%M:%S");

  return date.str();
}

void RecordingAutomation::update()
{
  if (!shouldTakeScreenshot && !takeVideo)
    return;

  if (shouldMoveToPosition)
  {
    std::string firstPosition =
        cameraPositionName.substr(0, cameraPositionName.find(","));
    std::string name = firstPosition.substr(0, cameraPositionName.find("_"));
    moveToCameraPosition(name);
    shouldMoveToPosition = false;
    return;
  }

  if (labellingCoordinator->haveLabelPositionsChanged())
  {
    unchangedCount = 0;
    startTime = std::chrono::high_resolution_clock::now();
  }
  else
  {
    unchangedCount++;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(endTime -
                                                                    startTime);
  if (unchangedCount > 10 && diff.count() > 1500)
  {
    int nextIndex = cameraPositionName.find(",");
    std::string name = cameraPositionName.substr(0, nextIndex);

    takeScreenshot(name);

    if (nextIndex > 0)
    {
      cameraPositionName = cameraPositionName.substr(nextIndex + 1);
      shouldMoveToPosition = true;
      shouldTakeScreenshot = true;
      return;
    }

    if (exitAfterScreenshot)
      exit(0);
  }
}

void RecordingAutomation::takeScreenshotOfNextFrame()
{
  shouldTakeScreenshot = true;
}

void RecordingAutomation::takeScreenshotOf(std::string cameraPositionName)
{
  this->cameraPositionName = cameraPositionName;
  shouldMoveToPosition = true;
  shouldTakeScreenshot = true;
}

void RecordingAutomation::takeScreenshotOfPositionAndExit(
    std::string cameraPositionName)
{
  exitAfterScreenshot = true;
  takeScreenshotOf(cameraPositionName);
}

void RecordingAutomation::startVideo(std::string positions)
{
  std::string filename =
      "video_" + nodes->getSceneName() + "_" + getTime() + ".mpeg";
  videoRecorder->createNewVideo(filename);
  videoRecorder->startRecording();
  cameraPositionName = positions;
  shouldMoveToPosition = true;
  takeVideo = true;
}

void RecordingAutomation::startVideoAndExit(std::string positions)
{
  exitAfterScreenshot = true;
  startVideo(positions);
}

void RecordingAutomation::takeScreenshot(std::string name)
{
  if (!shouldTakeScreenshot)
    return;

  std::string detail = name.size() ? name : getTime();
  std::string filename =
      "screenshot_" + nodes->getSceneName() + "_" + detail + ".png";
  std::vector<unsigned char> pixels(width * height * 4);
  gl->glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                   pixels.data());
  ImagePersister::flipAndSaveRGBA8I(pixels.data(), width, height, filename);
  shouldTakeScreenshot = false;
  qCWarning(recordingAutomationChan) << "Took screenshot:" << filename.c_str();
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
  float duration = takeVideo ? 4.0f : 1e-9f;
  cameraNode->getCamera()->startAnimation(target, duration);
}

