#ifndef TEST_TEST_H_

#define TEST_TEST_H_

#include <gtest/gtest.h>

#define EXPECT_Vector2f_NEAR(expected, actual, delta)                          \
  EXPECT_NEAR(expected.x(), actual.x(), delta);                                \
  EXPECT_NEAR(expected.y(), actual.y(), delta);

#define EXPECT_Vector3f_NEAR(expected, actual, delta)                          \
  EXPECT_NEAR(expected.x(), actual.x(), delta);                                \
  EXPECT_NEAR(expected.y(), actual.y(), delta);                                \
  EXPECT_NEAR(expected.z(), actual.z(), delta);

#define EXPECT_Vector4f_NEAR(expected, actual, delta)                          \
  EXPECT_NEAR(expected.x(), actual.x(), delta);                                \
  EXPECT_NEAR(expected.y(), actual.y(), delta);                                \
  EXPECT_NEAR(expected.z(), actual.z(), delta);                                \
  EXPECT_NEAR(expected.w(), actual.w(), delta);

#define EXPECT_Matrix4f_NEAR(expected, actual, delta)                          \
  EXPECT_NEAR(expected(0, 0), actual(0, 0), delta);                            \
  EXPECT_NEAR(expected(0, 1), actual(0, 1), delta);                            \
  EXPECT_NEAR(expected(0, 2), actual(0, 2), delta);                            \
  EXPECT_NEAR(expected(0, 3), actual(0, 3), delta);                            \
  EXPECT_NEAR(expected(1, 0), actual(1, 0), delta);                            \
  EXPECT_NEAR(expected(1, 1), actual(1, 1), delta);                            \
  EXPECT_NEAR(expected(1, 2), actual(1, 2), delta);                            \
  EXPECT_NEAR(expected(1, 3), actual(1, 3), delta);                            \
  EXPECT_NEAR(expected(2, 0), actual(2, 0), delta);                            \
  EXPECT_NEAR(expected(2, 1), actual(2, 1), delta);                            \
  EXPECT_NEAR(expected(2, 2), actual(2, 2), delta);                            \
  EXPECT_NEAR(expected(2, 3), actual(2, 3), delta);                            \
  EXPECT_NEAR(expected(3, 0), actual(3, 0), delta);                            \
  EXPECT_NEAR(expected(3, 1), actual(3, 1), delta);                            \
  EXPECT_NEAR(expected(3, 2), actual(3, 2), delta);                            \
  EXPECT_NEAR(expected(3, 3), actual(3, 3), delta);

template <class T>
void printVectorAsMatrix(T vector, int width, int height)
{
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; ++j)
      std::cout << vector[i * width + j] << " ";

    std::cout << std::endl;
  }
}

#endif  // TEST_TEST_H_
