#ifndef SRC_CAMERA_CONTROLLERS_H_

#define SRC_CAMERA_CONTROLLERS_H_

#include <memory>
#include "./camera.h"

class InvokeManager;
class CameraController;
class CameraRotationController;
class CameraZoomController;
class CameraMoveController;

/**
 * \brief Encapsulates all different camera controllers
 *
 */
class CameraControllers
{
 public:
  CameraControllers(std::shared_ptr<InvokeManager> invokeManager,
                    std::shared_ptr<Camera> camera);

  void update(std::shared_ptr<Camera> camera, double frameTime);

 private:
  std::shared_ptr<CameraController> cameraController;
  std::shared_ptr<CameraRotationController> cameraRotationController;
  std::shared_ptr<CameraZoomController> cameraZoomController;
  std::shared_ptr<CameraMoveController> cameraMoveController;
};

#endif  // SRC_CAMERA_CONTROLLERS_H_
