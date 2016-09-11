#ifndef SRC_RECORDING_AUTOMATION_H_

#define SRC_RECORDING_AUTOMATION_H_

#include <memory>
#include <string>
#include "./graphics/gl.h"

class LabellingCoordinator;
class Nodes;
class VideoRecorder;

/**
 * \brief
 *
 *
 */
class RecordingAutomation
{
 public:
  RecordingAutomation(
      std::shared_ptr<LabellingCoordinator> labellingCoordinator,
      std::shared_ptr<Nodes> nodes,
      std::shared_ptr<VideoRecorder> videoRecorder);

  void update();

  void initialize(Graphics::Gl *gl);

  void resize(int width, int height);

  void takeScreenshotOfNextFrame();

  void takeScreenshotOf(std::string cameraPositionName);

 private:
  std::shared_ptr<LabellingCoordinator> labellingCoordinator;
  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<VideoRecorder> videoRecorder;

  Graphics::Gl *gl;
  int width;
  int height;

  bool takeScreenshot;
  std::string cameraPositionName;

  int unchangedCount = 0;

  void moveToCameraPosition(std::string name);
};

#endif  // SRC_RECORDING_AUTOMATION_H_
