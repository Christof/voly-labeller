#include "./camera_zoom_controller.h"
#include <QDebug>
#include <QEvent>
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

void CameraZoomController::scroll(QEvent *event)
{
  qWarning() << event->type();
}

