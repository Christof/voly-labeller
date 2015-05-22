#include "./mouse_dragging_controller.h"
#include <QMoveEvent>

void MouseDraggingController::setFrameTime(double frameTime)
{
  this->frameTime = frameTime;
}

inline Eigen::Vector2f toEigen(const QPoint &pos)
{
  return Eigen::Vector2f(pos.x(), pos.y());
}

void MouseDraggingController::startDragging()
{
  start = true;
}

void MouseDraggingController::updateDragging(QEvent *event)
{
  auto mousePosition = toEigen(static_cast<QMouseEvent *>(event)->pos());

  if (start)
  {
    start = false;
    lastMousePosition = mousePosition;
    return;
  }

  Eigen::Vector2f diff = mousePosition - lastMousePosition;

  update(diff);

  lastMousePosition = mousePosition;
}

