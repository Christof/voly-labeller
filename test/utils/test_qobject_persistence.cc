#include "../test.h"
#include <Eigen/Core>
#include "../../src/utils/persister.h"

TEST(Test_QObjectPersistence, SaveAndLoadASimpleString)
{
  std::string text = "example text";

  Persister::save(text, "test.xml");

  EXPECT_EQ(text, Persister::load<std::string>("test.xml"));
}

TEST(Test_QObjectPersistence, SaveAndLoadAVector)
{
  Eigen::Vector3f vector(1.0f, 1.1f, 1.2f);

  Persister::save(vector, "test.xml");
  auto loaded = Persister::load<Eigen::Vector3f>("test.xml");
  EXPECT_NEAR(vector.x(), loaded.x(), 0.00001f);
}
