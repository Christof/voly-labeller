#ifndef TEST_TEST_H_

#define TEST_TEST_H_

#include <gtest/gtest.h>

#define EXPECT_Vector3f_NEAR(expected, actual, delta)                          \
  EXPECT_NEAR(expected.x(), actual.x(), delta);                                \
  EXPECT_NEAR(expected.y(), actual.y(), delta);                                \
  EXPECT_NEAR(expected.z(), actual.z(), delta);

#endif  // TEST_TEST_H_
