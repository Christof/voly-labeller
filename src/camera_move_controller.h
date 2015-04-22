#ifndef SRC_CAMERA_MOVE_CONTROLLER_H_

#define SRC_CAMERA_MOVE_CONTROLLER_H_

#include "./mouse_dragging_controller.h"

class Camera;

/**
 * \brief Controls movement of the camrea using mouse dragging with shift
 *
 * CameraMoveController::setFrameTime must be called each frame
 * to ensure a steady camera speed.
 */
class CameraMoveController : public MouseDraggingController
{
 public:
  explicit CameraMoveController(Camera &camera);

 protected:
  virtual void update(Eigen::Vector2f diff);

 private:
  Camera &camera;
  double speedFactor = 0.2;
};

#endif  // SRC_CAMERA_MOVE_CONTROLLER_H_
