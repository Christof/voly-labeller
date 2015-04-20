#include "./camera_rotation_controller.h"
#include <QCursor>
#include <QDebug>
#include "./eigen_qdebug.h"

CameraRotationController::CameraRotationController(Camera &camera)
  : camera(camera)
{
}

CameraRotationController::~CameraRotationController()
{
}

void CameraRotationController::setFrameTime(double frameTime)
{
  this->frameTime = frameTime;
}

void CameraRotationController::setRotate()
{
  lastMousePosition = Eigen::Vector2f(QCursor::pos().x(), QCursor::pos().y());
  qDebug() << "Start dragging at: " << lastMousePosition;
}

void CameraRotationController::updateRotate()
{
  auto mousePosition = Eigen::Vector2f(QCursor::pos().x(), QCursor::pos().y());
  double scaling = frameTime * speedFactor / camera.getPosition().norm();
  Eigen::Vector2f diff = scaling * (lastMousePosition - mousePosition);

  camera.changeAzimuth(-atan(diff.x()));
  camera.changeDeclination(atan(diff.y()));

  lastMousePosition = mousePosition;
}

void CameraRotationController::endRotate()
{
  qDebug() << "Stop dragging at: " << QCursor::pos();
}

