#ifndef SRC_CAMERA_ZOOM_CONTROLLER_H_

#define SRC_CAMERA_ZOOM_CONTROLLER_H_

#include <QObject>
#include <Eigen/Core>
#include "./camera.h"

/**
 * \brief Controls zoom of the camrea using mouse dragging with ctrl
 *
 * It is enabled via the state machine, which calls the slots.
 * The mouse position is gathered using QCursor.
 *
 * CameraZoomController::setFrameTime must be called each frame
 * to ensure a steady camera speed.
 */
class CameraZoomController : public QObject
{
  Q_OBJECT
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit CameraZoomController(Camera &camera);
  virtual ~CameraZoomController();

  void setFrameTime(double frameTime);

 public slots:
  void setRotate();
  void updateRotate();

 private:
  Camera &camera;
  Eigen::Vector2f lastMousePosition;
  double frameTime;
  double speedFactor = 1;
};

#endif  // SRC_CAMERA_ZOOM_CONTROLLER_H_
