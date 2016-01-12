#ifndef SRC_CAMERA_ROTATION_CONTROLLER_H_

#define SRC_CAMERA_ROTATION_CONTROLLER_H_

#include "./mouse_dragging_controller.h"

class Camera;

/**
 * \brief Controls rotation of the camrea using mouse dragging
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
  double speedFactor = 0.04;
};

#endif  // SRC_CAMERA_ROTATION_CONTROLLER_H_
