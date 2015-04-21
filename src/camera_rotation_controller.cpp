#include "./camera_rotation_controller.h"
#include <QCursor>
#include "./camera.h"

CameraRotationController::CameraRotationController(Camera &camera)
  : camera(camera)
{
}

void CameraRotationController::update(Eigen::Vector2f diff)
{
  double scaling = frameTime * speedFactor / camera.getPosition().norm();
  Eigen::Vector2f delta = scaling * diff;

  camera.changeAzimuth(atan(delta.x()));
  camera.changeDeclination(-atan(delta.y()));
}

