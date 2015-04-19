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
      Eigen::Vector2f(window.mousePosition.x(), window.mousePosition.y());
  qDebug() << "Start dragging at: " << mousePositionStart;
}

void CameraRotationController::updateRotate()
{
  qDebug() << "Update rotate";
}
