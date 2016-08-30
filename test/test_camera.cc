#include "./test.h"
#include "../src/camera.h"

TEST(Test_Camera, ConstructorFromMatricesWithDefaultValues)
{
  Camera expected;
  expected.resize(1, 1);

  Camera camera(expected.getViewMatrix(), expected.getProjectionMatrix(),
                expected.getOrigin());

  EXPECT_Vector3f_NEAR(expected.getPosition(), camera.getPosition(), 1e-5f);
  EXPECT_NEAR(expected.getRadius(), camera.getRadius(), 1e-5f);

  // just to recalculate the view matrix from the angles and test them
  camera.changeAzimuth(0);

  EXPECT_Matrix4f_NEAR(expected.getViewMatrix(), camera.getViewMatrix(), 1e-5f);
  EXPECT_Vector3f_NEAR(expected.getPosition(), camera.getPosition(), 1e-5f);
  EXPECT_TRUE(camera.needsResizing());

  camera.resize(1, 1);

  EXPECT_Matrix4f_NEAR(expected.getProjectionMatrix(),
                       camera.getProjectionMatrix(), 1e-5f);
}

TEST(Test_Camera, ConstructorFromMatricesAfterRotation)
{
  Camera expected;
  expected.resize(1, 1);
  expected.changeAzimuth(M_PI * 0.25f);

  Camera camera(expected.getViewMatrix(), expected.getProjectionMatrix(),
                expected.getOrigin());

  EXPECT_Vector3f_NEAR(expected.getPosition(), camera.getPosition(), 1e-5f);
  EXPECT_NEAR(expected.getRadius(), camera.getRadius(), 1e-5f);

  // just to recalculate the view matrix from the angles and test them
  camera.changeAzimuth(0);

  EXPECT_Matrix4f_NEAR(expected.getViewMatrix(), camera.getViewMatrix(), 1e-5f);
  EXPECT_Matrix4f_NEAR(expected.getProjectionMatrix(),
                       camera.getProjectionMatrix(), 1e-5f);
  EXPECT_Vector3f_NEAR(expected.getPosition(), camera.getPosition(), 1e-5f);
  EXPECT_TRUE(camera.needsResizing());
}

TEST(Test_Camera, AnimationToTheSamePosition)
{
  Camera camera;
  Eigen::Matrix4f viewMatrix = camera.getViewMatrix();
  camera.startAnimation(viewMatrix, 1.0f);

  camera.updateAnimation(1.0f);

  EXPECT_Matrix4f_NEAR(viewMatrix, camera.getViewMatrix(), 1e-5f);
}
