#ifndef SRC_CAMERA_ROTATION_CONTROLLER_H_

#define SRC_CAMERA_ROTATION_CONTROLLER_H_

#include <QObject>
#include <QCursor>
#include <Eigen/Core>
#include "./window.h"

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

  CameraRotationController(Window &window);
  virtual ~CameraRotationController();

 public slots:
  void setRotate();
  void updateRotate();
  void endRotate();

 private:
  QCursor cursor;
  Window &window;
  Eigen::Vector2f mousePositionStart;
};

#endif  // SRC_CAMERA_ROTATION_CONTROLLER_H_
