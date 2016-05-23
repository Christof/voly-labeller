#include "./picking_controller.h"
#include <QMouseEvent>
#include <QDebug>
#include "./math/eigen.h"
#include "./scene.h"

PickingController::PickingController(std::shared_ptr<Scene> scene)
  : scene(scene)
{
}

void PickingController::startPicking(Label label)
{
  this->label = label;
}

void PickingController::pick(QEvent *event)
{
  auto mouseEvent = static_cast<QMouseEvent *>(event);
  if (mouseEvent->button() != Qt::MouseButton::LeftButton)
    return;

  auto position = toEigen(mouseEvent->localPos());
  scene->pick(label.id, position);
}

void PickingController::pickRotationPosition(QEvent *event)
{
  auto mouseEvent = static_cast<QMouseEvent *>(event);
  qInfo() << "In pickRotationPosition" << mouseEvent;

  auto position = toEigen(mouseEvent->localPos());
  scene->pick(-1, position);
}
