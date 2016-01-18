#include "./camera_controllers.h"
#include "./input/invoke_manager.h"
#include "./camera_controller.h"
#include "./camera_rotation_controller.h"
#include "./camera_zoom_controller.h"
#include "./camera_move_controller.h"

CameraControllers::CameraControllers(
    std::shared_ptr<InvokeManager> invokeManager, Camera &camera)
{
  cameraController = std::make_shared<CameraController>(camera);
  cameraRotationController = std::make_shared<CameraRotationController>(camera);
  cameraZoomController = std::make_shared<CameraZoomController>(camera);
  cameraMoveController = std::make_shared<CameraMoveController>(camera);

  invokeManager->addHandler("cam", cameraController.get());
  invokeManager->addHandler("cameraRotation", cameraRotationController.get());
  invokeManager->addHandler("cameraZoom", cameraZoomController.get());
  invokeManager->addHandler("cameraMove", cameraMoveController.get());
}

void CameraControllers::update(Camera &camera, double frameTime)
{
  cameraController->update(camera, frameTime);
  cameraRotationController->update(camera, frameTime);
  cameraZoomController->update(camera, frameTime);
  cameraMoveController->update(camera, frameTime);
}

