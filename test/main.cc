#include <gtest/gtest.h>

#include <QLoggingCategory>

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  QLoggingCategory::setFilterRules("*=false");

  return RUN_ALL_TESTS();
}
