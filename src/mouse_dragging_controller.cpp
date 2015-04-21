#include "./mouse_dragging_controller.h"
#include <QCursor>

void MouseDraggingController::setFrameTime(double frameTime)
{
  this->frameTime = frameTime;
}

void MouseDraggingController::startDragging()
{
  lastMousePosition = Eigen::Vector2f(QCursor::pos().x(), QCursor::pos().y());
}

void MouseDraggingController::updateDragging()
{
  auto mousePosition = Eigen::Vector2f(QCursor::pos().x(), QCursor::pos().y());
  Eigen::Vector2f diff = mousePosition - lastMousePosition;

  update(diff);

  lastMousePosition = mousePosition;
}

