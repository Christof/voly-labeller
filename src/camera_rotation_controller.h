#ifndef SRC_CAMERA_ROTATION_CONTROLLER_H_

#define SRC_CAMERA_ROTATION_CONTROLLER_H_

#include <QObject>
#include <Eigen/Core>
#include "./camera.h"

/**
 * \brief
 *
 *
 */
class CameraRotationController : public QObject
{
  Q_OBJECT
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraRotationController(Camera &camera);
  virtual ~CameraRotationController();

 public slots:
  void setRotate();
  void updateRotate();
  void endRotate();

 private:
  Camera &camera;
  Eigen::Vector2f mousePositionStart;
};

#endif  // SRC_CAMERA_ROTATION_CONTROLLER_H_
