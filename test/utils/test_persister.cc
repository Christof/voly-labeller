#include "../test.h"
#include <Eigen/Core>
#include "../../src/utils/persister.h"
#include "../../src/label.h"

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
  EXPECT_NEAR(vector.y(), loaded.y(), 0.00001f);
  EXPECT_NEAR(vector.z(), loaded.z(), 0.00001f);
}

TEST(Test_QObjectPersistence, SaveAndLoadALabel)
{
  Label label(2, "my label 2", Eigen::Vector3f(1, 2, 3));

  Persister::save(label, "test.xml");
  auto loaded = Persister::load<Label>("test.xml");
  EXPECT_EQ(label.id, loaded.id);
  EXPECT_EQ(label.text, loaded.text);
  EXPECT_NEAR(label.anchorPosition.x(), loaded.anchorPosition.x(), 0.00001f);
  EXPECT_NEAR(label.anchorPosition.y(), loaded.anchorPosition.y(), 0.00001f);
  EXPECT_NEAR(label.anchorPosition.z(), loaded.anchorPosition.z(), 0.00001f);
}
