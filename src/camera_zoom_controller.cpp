#include "./camera_zoom_controller.h"
#include <QCursor>
#include "./camera.h"

CameraZoomController::CameraZoomController(Camera &camera) : camera(camera)
{
}

void CameraZoomController::update(Eigen::Vector2f diff)
{
  double scaling = frameTime * speedFactor;
  Eigen::Vector2f delta = scaling * diff;

  camera.changeRadius(delta.y());
}

