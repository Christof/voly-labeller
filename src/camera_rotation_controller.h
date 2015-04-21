#ifndef SRC_CAMERA_ROTATION_CONTROLLER_H_

#define SRC_CAMERA_ROTATION_CONTROLLER_H_

#include <QObject>
#include <Eigen/Core>
#include "./camera.h"

/**
 * \brief Controls rotation of the camrea using mouse dragging
 *
 * It is enabled via the state machine, which calls the slots.
 * The mouse position is gathered using QCursor.
 *
 * CameraRotationController::setFrameTime must be called each frame
 * to ensure a steady camera speed.
 */
class CameraRotationController : public QObject
{
  Q_OBJECT
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit CameraRotationController(Camera &camera);
  virtual ~CameraRotationController();

  void setFrameTime(double frameTime);

 public slots:
  void startDragging();
  void updateDragging();

 private:
  Camera &camera;
  Eigen::Vector2f lastMousePosition;
  double frameTime;
  double speedFactor = 0.5;
};

#endif  // SRC_CAMERA_ROTATION_CONTROLLER_H_
