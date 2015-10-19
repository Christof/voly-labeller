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

void CameraControllers::update(double frameTime)
{
  cameraController->setFrameTime(frameTime);
  cameraRotationController->setFrameTime(frameTime);
  cameraZoomController->setFrameTime(frameTime);
  cameraMoveController->setFrameTime(frameTime);
}

