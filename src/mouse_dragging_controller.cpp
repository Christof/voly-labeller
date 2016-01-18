#include "./mouse_dragging_controller.h"
#include <QMouseEvent>
#include "./camera.h"

MouseDraggingController::MouseDraggingController(Camera &camera,
                                                 double speedFactor)
    : camera(camera), speedFactor(speedFactor)
{
}

void MouseDraggingController::update(Camera &camera, double frameTime)
{
  this->camera = camera;
  this->frameTime = frameTime;
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

  updateFromDiff(diff);

  lastMousePosition = mousePosition;
}

