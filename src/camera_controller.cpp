#include "./camera_controller.h"
#include <QDebug>

CameraController::CameraController(Camera &camera) : camera(camera)
{
}

CameraController::~CameraController()
{
}

void CameraController::setFrameTime(double frameTime)
{
  this->frameTime = frameTime;
}

void CameraController::moveForward()
{
  camera.moveForward(frameTime * cameraSpeed);
}

void CameraController::moveBackward()
{
  camera.moveBackward(frameTime * cameraSpeed);
}

void CameraController::strafeLeft()
{
  camera.strafeLeft(frameTime * cameraSpeed);
}

void CameraController::strafeRight()
{
  camera.strafeRight(frameTime * cameraSpeed);
}

