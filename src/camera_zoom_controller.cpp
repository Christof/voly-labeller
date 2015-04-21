#include "./camera_zoom_controller.h"
#include <QCursor>

CameraZoomController::CameraZoomController(Camera &camera) : camera(camera)
{
}

CameraZoomController::~CameraZoomController()
{
}

void CameraZoomController::setFrameTime(double frameTime)
{
  this->frameTime = frameTime;
}

void CameraZoomController::startDragging()
{
  lastMousePosition = Eigen::Vector2f(QCursor::pos().x(), QCursor::pos().y());
}

void CameraZoomController::updateDragging()
{
  auto mousePosition = Eigen::Vector2f(QCursor::pos().x(), QCursor::pos().y());
  double scaling = frameTime * speedFactor;
  Eigen::Vector2f diff = scaling * (lastMousePosition - mousePosition);

  camera.changeRadius(diff.y());

  lastMousePosition = mousePosition;
}

