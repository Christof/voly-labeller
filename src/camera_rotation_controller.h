#ifndef SRC_CAMERA_ROTATION_CONTROLLER_H_

#define SRC_CAMERA_ROTATION_CONTROLLER_H_

#include "./mouse_dragging_controller.h"

class Camera;

/**
 * \brief Controls rotation of the camrea using mouse dragging
 *
 * It is enabled via the state machine, which calls the slots.
 * The mouse position is gathered using QCursor.
 *
 * CameraRotationController::setFrameTime must be called each frame
 * to ensure a steady camera speed.
 */
class CameraRotationController : public MouseDraggingController
{
 public:
  explicit CameraRotationController(Camera &camera);

 protected:
  virtual void update(Eigen::Vector2f diff);

 private:
  Camera &camera;
  double speedFactor = 0.5;
};

#endif  // SRC_CAMERA_ROTATION_CONTROLLER_H_
