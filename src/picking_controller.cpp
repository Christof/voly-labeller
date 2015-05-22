#include "./picking_controller.h"
#include <QMouseEvent>
#include "./math/eigen.h"
#include "./scene.h"

PickingController::PickingController(std::shared_ptr<Scene> scene)
  : scene(scene)
{
}

void PickingController::pick(QEvent *event)
{
  auto mouseEvent =static_cast<QMouseEvent*>(event);
  if (mouseEvent->button() != Qt::MouseButton::LeftButton)
    return;

  auto position = toEigen(mouseEvent->localPos());
  scene->pick(position);
}
