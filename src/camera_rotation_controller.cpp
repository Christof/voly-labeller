#include "./camera_rotation_controller.h"
#include <QCursor>
#include "./camera.h"

CameraRotationController::CameraRotationController(
    std::shared_ptr<Camera> camera)
  : MouseDraggingController(camera, 0.04)
{
}

void CameraRotationController::updateFromDiff(Eigen::Vector2f diff)
{
  double scaling = frameTime * speedFactor;
  Eigen::Vector2f delta = scaling * diff;

  camera->changeAzimuth(atan(delta.x()));
  camera->changeDeclination(-atan(delta.y()));
}

