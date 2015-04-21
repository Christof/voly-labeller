#ifndef SRC_CAMERA_ZOOM_CONTROLLER_H_

#define SRC_CAMERA_ZOOM_CONTROLLER_H_

#include "./mouse_dragging_controller.h"

class Camera;

/**
 * \brief Controls zoom of the camrea using mouse dragging with ctrl
 *
 * It is enabled via the state machine, which calls the slots.
 * The mouse position is gathered using QCursor.
 *
 * CameraZoomController::setFrameTime must be called each frame
 * to ensure a steady camera speed.
 */
class CameraZoomController : public MouseDraggingController
{
 public:
  explicit CameraZoomController(Camera &camera);
  void updateDragging();

 protected:
  virtual void update(Eigen::Vector2f diff);

 private:
  Camera &camera;
  double speedFactor = 1;
};

#endif  // SRC_CAMERA_ZOOM_CONTROLLER_H_
