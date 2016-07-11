#ifndef SRC_CAMERA_ROTATION_CONTROLLER_H_

#define SRC_CAMERA_ROTATION_CONTROLLER_H_

#include "./mouse_dragging_controller.h"

class Camera;

/**
 * \brief Controls rotation of the camrea using mouse dragging
 *
 * CameraRotationController::setFrameTime must be called each frame
 * to ensure a steady camera speed.
 */
class CameraRotationController : public MouseDraggingController
{
 public:
  explicit CameraRotationController(std::shared_ptr<Camera> camera);

 protected:
  virtual void updateFromDiff(Eigen::Vector2f diff);
  virtual void startOfDragging(Eigen::Vector2f startMousePosition);

 private:
  Eigen::Vector3f convertXY(Eigen::Vector2f vec);
  Eigen::Matrix4f applyTranslationMatrix(bool reverse);
  void applyRotationMatrix();
  void stopRotation();

  Eigen::Vector3f startRotationVector;
  Eigen::Vector3f currentRotationVector;

  Eigen::Vector3f startPosition;

  bool isRotating = false;
  float transX;
  float transY;

  float currentTransX;
  float currentTransY;

  float startTransX;
  float startTransY;

  Eigen::Matrix4f startMatrix = Eigen::Matrix4f::Identity();
};

#endif  // SRC_CAMERA_ROTATION_CONTROLLER_H_
