#include "./mouse_dragging_controller.h"
#include <QMouseEvent>
#include "./camera.h"
#include <iostream>

MouseDraggingController::MouseDraggingController(std::shared_ptr<Camera> camera,
                                                 double speedFactor)
  : camera(camera), speedFactor(speedFactor)
{
}

void MouseDraggingController::update(std::shared_ptr<Camera> camera,
                                     double frameTime)
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
  mousePosition = toEigen(static_cast<QMouseEvent *>(event)->pos());
  std::cout << "Mouse position: " << mousePosition.transpose() << std::endl;

  if (start)
  {
    start = false;
    lastMousePosition = mousePosition;
    startOfDragging(mousePosition);
    return;
  }

  Eigen::Vector2f diff = mousePosition - lastMousePosition;

  updateFromDiff(diff);

  lastMousePosition = mousePosition;
}

