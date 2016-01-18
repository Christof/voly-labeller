#include "./camera_zoom_controller.h"
#include <QDebug>
#include <QEvent>
#include <QWheelEvent>
#include "./camera.h"

CameraZoomController::CameraZoomController(Camera &camera)
  : MouseDraggingController(camera, 0.1)
{
}

void CameraZoomController::updateFromDiff(Eigen::Vector2f diff)
{
  double scaling = frameTime * speedFactor;
  Eigen::Vector2f delta = scaling * diff;

  camera.changeRadius(delta.y());
}

void CameraZoomController::wheelZoom(QEvent *event)
{
  QWheelEvent *wheelEvent = static_cast<QWheelEvent *>(event);

  double scaling = -0.02f * frameTime * speedFactor * camera.getRadius();
  float delta = scaling * wheelEvent->angleDelta().y();

  camera.changeRadius(delta);
}

