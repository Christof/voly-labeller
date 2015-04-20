#include "./camera_rotation_controller.h"
#include <QDebug>
#include "./eigen_qdebug.h"

CameraRotationController::CameraRotationController(Window &window) : window(window)
{
}

CameraRotationController::~CameraRotationController()
{
}

void CameraRotationController::setRotate()
{
  mousePositionStart =
      Eigen::Vector2f(cursor.pos().x(), cursor.pos().y());
  qDebug() << "Start dragging at: " << cursor.pos();
  cursor.setShape(Qt::CursorShape::ClosedHandCursor);
  window.setCursor(cursor);
}

void CameraRotationController::updateRotate()
{
  auto position = cursor.pos();
  qDebug() << "Update rotate" << position;
}

void CameraRotationController::endRotate()
{
  qDebug() << "Stop dragging at: " << cursor.pos();
  cursor.setShape(Qt::CursorShape::ArrowCursor);
  window.setCursor(cursor);
}
