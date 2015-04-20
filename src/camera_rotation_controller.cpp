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

void CameraRotationController::setRotate()
{
  mousePositionStart = Eigen::Vector2f(QCursor::pos().x(), QCursor::pos().y());
  qDebug() << "Start dragging at: " << mousePositionStart;
}

void CameraRotationController::updateRotate()
{
  qDebug() << "Update rotate" << QCursor::pos();
}

void CameraRotationController::endRotate()
{
  qDebug() << "Stop dragging at: " << QCursor::pos();
}
