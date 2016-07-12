#include "./camera_rotation_controller.h"
#include <QCursor>
#include "./camera.h"
#include <iostream>

CameraRotationController::CameraRotationController(
    std::shared_ptr<Camera> camera)
  : MouseDraggingController(camera, 0.04)
{
}

Eigen::Vector3f CameraRotationController::convertXY(Eigen::Vector2f vec)
{
  const float width = 1000;
  const float height = 1000;
  Eigen::Vector2f center(0.5f * width, 0.5f * height);
  vec = Eigen::Vector2f(vec.x() - center.x(), center.y() - vec.y());

  float normSquared = vec.dot(vec);
  float ballRadius = 500;
  float radiusSquared = ballRadius * ballRadius;
  if (normSquared > radiusSquared)
  {
    float factor = 1.0f / sqrt(normSquared);
    return factor * Eigen::Vector3f(vec.x(), vec.y(), 0);
  }

  return Eigen::Vector3f(vec.x(), vec.y(), sqrt(radiusSquared - normSquared));
}

void
CameraRotationController::startOfDragging(Eigen::Vector2f startMousePosition)
{
  startRotationVector = convertXY(startMousePosition);
  startRotationVector.normalize();

  currentRotationVector = startRotationVector;
  isRotating = true;
  startMatrix = camera->getViewMatrix();

  startPosition = camera->getPosition();
}

void CameraRotationController::updateFromDiff(Eigen::Vector2f diff)
{
  currentRotationVector = convertXY(mousePosition);

  currentRotationVector.normalize();
  /*
  double scaling = frameTime * speedFactor;
  Eigen::Vector2f delta = scaling * diff;
  // delta.x() = 0;

  camera->rotateAroundOrbit(delta);
  // camera->changeAzimuth(atan(delta.x()));
  // camera->changeDeclination(-atan(delta.y()));
   */
  applyRotationMatrix();
  startOfDragging(mousePosition);
}

void CameraRotationController::applyRotationMatrix()
{
  if (isRotating)
  {  // Do some rotation according to start and current rotation vectors
    /*
    std::cerr << currentRotationVector.transpose() << "\t"
              << startRotationVector.transpose() << std::endl;
              */
    if ((currentRotationVector - startRotationVector).norm() > 1E-6)
    {
      Eigen::Vector3f rotationAxis =
          currentRotationVector.cross(startRotationVector);
      rotationAxis.normalize();

      double val = currentRotationVector.dot(startRotationVector);
      val = val > (1 - 1E-10) ? 1.0 : val;
      double rotationAngle = acos(val);

      // rotate around the current position
      /*
      applyTranslationMatrix(true);
      glRotatef(rotationAngle * 2, -rotationAxis.x(), -rotationAxis.y(),
                -rotationAxis.z());
      applyTranslationMatrix(false);
      */
      /*
      Eigen::Matrix4f view =
          applyTranslationMatrix(true) *
          Eigen::Affine3f(Eigen::AngleAxisf(rotationAngle, -rotationAxis)) *
          applyTranslationMatrix(false);
          */
      Eigen::Matrix4f view =
          Eigen::Affine3f(Eigen::Translation3f(-startPosition) *
                          Eigen::AngleAxisf(rotationAngle, -rotationAxis) *
                          Eigen::Translation3f(startPosition)).matrix();


      camera->setViewMatrix(startMatrix * view);
    }
  }
  // glMultMatrixf(startMatrix);
}

/**
 * \ingroup GLVisualization
* Stop the current rotation and prepare for a new click-then-drag event
*
**/
void CameraRotationController::stopRotation()
{
  // glMatrixMode(GL_MODELVIEW);
  // glLoadIdentity();
  applyRotationMatrix();
  // set the current matrix as the permanent one
  // glGetFloatv(GL_MODELVIEW_MATRIX, startMatrix);
  startMatrix = camera->getViewMatrix();
  isRotating = false;
}

/**
 * Apply the translation matrix to the current transformation (zoom factor)
 **/
Eigen::Matrix4f CameraRotationController::applyTranslationMatrix(bool reverse)
{
  const float TRANSLATION_FACTOR = 0.01f;
  float factor = (reverse ? -1.0f : 1.0f);
  float tx = transX + (currentTransX - startTransX) * TRANSLATION_FACTOR;
  float ty = transY + (currentTransY - startTransY) * TRANSLATION_FACTOR;
  // glTranslatef(factor * tx, factor * (-ty), 0);

  return Eigen::Affine3f(Eigen::Translation3f(factor * tx, factor * -ty, 0))
      .matrix();
}

