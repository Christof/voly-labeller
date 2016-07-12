#include <QMouseEvent>
#include <Eigen/Geometry>
#include "./test.h"
#include "../src/camera_rotation_controller.h"
#include "../src/camera.h"

TEST(Test_CameraRotationController, DragUpCausesRotationAroundXAxis)
{
  auto camera = std::make_shared<Camera>();
  CameraRotationController controller(camera);
  camera->setViewMatrix(Eigen::Matrix4f::Identity());

  controller.startDragging();
  QMouseEvent startEvent(QEvent::MouseMove, QPointF(500, 500),
                         Qt::MouseButton::NoButton, Qt::LeftButton,
                         Qt::NoModifier);
  controller.updateDragging(&startEvent);

  QMouseEvent updateEvent(QEvent::MouseMove, QPointF(500, 750),
                          Qt::MouseButton::NoButton, Qt::LeftButton,
                          Qt::NoModifier);
  controller.updateDragging(&updateEvent);

  Eigen::AngleAxisf rotation;
  rotation.fromRotationMatrix(camera->getViewMatrix().block<3, 3>(0, 0));

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1, 0, 0), rotation.axis(), 1e-6f);
  EXPECT_NEAR(0.5235989f, rotation.angle(), 1e-6f);
}

TEST(Test_CameraRotationController, DragRightCausesRotationAroundYAxis)
{
  auto camera = std::make_shared<Camera>();
  CameraRotationController controller(camera);
  camera->setViewMatrix(Eigen::Matrix4f::Identity());

  controller.startDragging();
  QMouseEvent startEvent(QEvent::MouseMove, QPointF(500, 500),
                         Qt::MouseButton::NoButton, Qt::LeftButton,
                         Qt::NoModifier);
  controller.updateDragging(&startEvent);

  QMouseEvent updateEvent(QEvent::MouseMove, QPointF(750, 500),
                          Qt::MouseButton::NoButton, Qt::LeftButton,
                          Qt::NoModifier);
  controller.updateDragging(&updateEvent);

  Eigen::AngleAxisf rotation;
  rotation.fromRotationMatrix(camera->getViewMatrix().block<3, 3>(0, 0));

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0, 1, 0), rotation.axis(), 1e-6f);
  EXPECT_NEAR(0.5235989f, rotation.angle(), 1e-6f);
}

TEST(Test_CameraRotationController, DragUpAndRight)
{
  auto camera = std::make_shared<Camera>();
  CameraRotationController controller(camera);
  camera->setViewMatrix(Eigen::Matrix4f::Identity());

  controller.startDragging();
  QMouseEvent startEvent(QEvent::MouseMove, QPointF(500, 500),
                         Qt::MouseButton::NoButton, Qt::LeftButton,
                         Qt::NoModifier);
  controller.updateDragging(&startEvent);

  QMouseEvent updateEvent(QEvent::MouseMove, QPointF(750, 750),
                          Qt::MouseButton::NoButton, Qt::LeftButton,
                          Qt::NoModifier);
  controller.updateDragging(&updateEvent);

  Eigen::AngleAxisf rotation;
  rotation.fromRotationMatrix(camera->getViewMatrix().block<3, 3>(0, 0));

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1, 1, 0).normalized(), rotation.axis(),
                       1e-6f);
  EXPECT_NEAR(0.785398f, rotation.angle(), 1e-6f);
}

TEST(Test_CameraRotationController, DragUpAndAfterwardsRight)
{
  auto camera = std::make_shared<Camera>();
  CameraRotationController controller(camera);
  camera->setViewMatrix(Eigen::Matrix4f::Identity());

  controller.startDragging();
  QMouseEvent startEvent(QEvent::MouseMove, QPointF(500, 500),
                         Qt::MouseButton::NoButton, Qt::LeftButton,
                         Qt::NoModifier);
  controller.updateDragging(&startEvent);

  QMouseEvent updateEvent(QEvent::MouseMove, QPointF(750, 500),
                          Qt::MouseButton::NoButton, Qt::LeftButton,
                          Qt::NoModifier);
  controller.updateDragging(&updateEvent);

  QMouseEvent updateEvent2(QEvent::MouseMove, QPointF(750, 750),
                           Qt::MouseButton::NoButton, Qt::LeftButton,
                           Qt::NoModifier);
  controller.updateDragging(&updateEvent2);

  Eigen::AngleAxisf rotation;
  rotation.fromRotationMatrix(camera->getViewMatrix().block<3, 3>(0, 0));

  EXPECT_Vector3f_NEAR(
      Eigen::Vector3f(0.644109010f, 0.745209157f, -0.17258849f),
      rotation.axis(), 1e-6f);
  EXPECT_NEAR(0.7980352f, rotation.angle(), 1e-6f);
}
