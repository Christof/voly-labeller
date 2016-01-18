#include "./camera_move_controller.h"
#include "./camera.h"

CameraMoveController::CameraMoveController(std::shared_ptr<Camera> camera)
  : MouseDraggingController(camera, 0.2)
{
}

void CameraMoveController::updateFromDiff(Eigen::Vector2f diff)
{
  Eigen::Vector2f delta = -frameTime * speedFactor * diff;

  camera->strafe(delta.x());
  camera->moveVertical(delta.y());
}

