#ifndef SRC_CAMERA_CONTROLLER_H_

#define SRC_CAMERA_CONTROLLER_H_

#include <QObject>
#include "./camera.h"

/**
 * \brief
 *
 *
 */
class CameraController : public QObject
{
  Q_OBJECT
 public:
  explicit CameraController(Camera &camera);
  virtual ~CameraController();

  void setFrameTime(double frameTime);

 public slots:
  void moveForward();
  void moveBackward();
  void strafeLeft();
  void strafeRight();
  void azimuthLeft();
  void azimuthRight();
  void increaseDeclination();
  void decreaseDeclination();

 private:
  Camera &camera;
  double cameraSpeed = 10.0f;
  double frameTime;
};

#endif  // SRC_CAMERA_CONTROLLER_H_
