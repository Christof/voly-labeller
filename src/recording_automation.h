#ifndef SRC_RECORDING_AUTOMATION_H_

#define SRC_RECORDING_AUTOMATION_H_

#include <memory>
#include <functional>
#include <string>
#include <chrono>
#include "./graphics/gl.h"

class LabellingCoordinator;
class Nodes;
class VideoRecorder;

/**
 * \brief Automation for taking a screenshot, with support for moving to a
 * camera position and wait for the labels to have reached their position
 *
 * #initialize and #resize must be called at startup. Each frame #update must be
 * called.
 */
class RecordingAutomation
{
 public:
  RecordingAutomation(
      std::shared_ptr<LabellingCoordinator> labellingCoordinator,
      std::shared_ptr<Nodes> nodes,
      std::shared_ptr<VideoRecorder> videoRecorder,
      std::function<void()> quit);

  void update();

  void initialize(Graphics::Gl *gl);

  void resize(int width, int height);

  void takeScreenshotOfNextFrame();

  void takeScreenshotOf(std::string cameraPositionName);
  void takeScreenshotOfPositionAndExit(std::string cameraPositionName);

  void startVideo(std::string positions);
  void startVideoAndExit(std::string positions);

  void startMovement(std::string positions);
  void startMovementAndExit(std::string positions);

 private:
  std::shared_ptr<LabellingCoordinator> labellingCoordinator;
  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<VideoRecorder> videoRecorder;
  std::function<void()> quit;

  Graphics::Gl *gl;
  int width;
  int height;

  bool shouldTakeScreenshot = false;
  bool takeVideo = false;
  bool shouldMoveToPosition = false;
  std::string cameraPositionName;
  bool exitAfterScreenshot = false;
  std::string remainingVideoSteps;

  int unchangedCount = 0;
  std::chrono::high_resolution_clock::time_point startTime;

  void takeScreenshot(std::string name);
  void moveToCameraPosition(std::string name);
};

#endif  // SRC_RECORDING_AUTOMATION_H_
