#ifndef SRC_RECORDING_AUTOMATION_CONTROLLER_H_

#define SRC_RECORDING_AUTOMATION_CONTROLLER_H_

#include <QObject>
#include <memory>
#include "./graphics/gl.h"

class LabellingCoordinator;
class VideoRecorder;

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
      std::shared_ptr<LabellingCoordinator> labellingCoordinator,
      std::shared_ptr<VideoRecorder> videoRecorder);

  void update();

  void initialize(Graphics::Gl *gl);

  void resize(int width, int height);

 public slots:
  void takeScreenshotOfNextFrame();

 private:
  std::shared_ptr<LabellingCoordinator> labellingCoordinator;
  std::shared_ptr<VideoRecorder> videoRecorder;

  Graphics::Gl *gl;
  int width;
  int height;

  bool takeScreenshot;
};

#endif  // SRC_RECORDING_AUTOMATION_CONTROLLER_H_
