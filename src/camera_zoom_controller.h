#ifndef SRC_CAMERA_ZOOM_CONTROLLER_H_

#define SRC_CAMERA_ZOOM_CONTROLLER_H_

#include "./mouse_dragging_controller.h"

class Camera;

/**
 * \brief Controls zoom of the camrea using mouse dragging with ctrl
 *
 * CameraZoomController::setFrameTime must be called each frame
 * to ensure a steady camera speed.
 */
class CameraZoomController : public MouseDraggingController
{
  Q_OBJECT

 public:
  explicit CameraZoomController(Camera &camera);

 public slots:
  void wheelZoom(QEvent *event);

 protected:
  virtual void update(Eigen::Vector2f diff);

 private:
  Camera &camera;
  double speedFactor = 1;
};

#endif  // SRC_CAMERA_ZOOM_CONTROLLER_H_
