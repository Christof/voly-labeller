#include "./camera_move_controller.h"
#include "./camera.h"

CameraMoveController::CameraMoveController(Camera &camera) : camera(camera)
{
}

void CameraMoveController::update(Eigen::Vector2f diff)
{
  Eigen::Vector2f delta = -frameTime * speedFactor * diff;

  camera.strafe(delta.x());
  camera.moveVertical(delta.y());
}

