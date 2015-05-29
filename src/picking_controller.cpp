#include "./picking_controller.h"
#include <QMouseEvent>
#include "./math/eigen.h"
#include "./scene.h"
#include "./label.h"

PickingController::PickingController(std::shared_ptr<Scene> scene)
  : scene(scene)
{
}

void PickingController::startPicking(Label *label)
{
  this->label = label;
}

void PickingController::pick(QEvent *event)
{
  auto mouseEvent = static_cast<QMouseEvent *>(event);
  if (mouseEvent->button() != Qt::MouseButton::LeftButton)
    return;

  auto position = toEigen(mouseEvent->localPos());
  scene->pick(position, std::bind(&PickingController::pickedPosition, this,
                                  std::placeholders::_1));
}

void PickingController::pickedPosition(Eigen::Vector3f position)
{
  label->anchorPosition = position;
}
