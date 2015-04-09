#include "../test.h"
#include <Eigen/Core>
#include "../../src/utils/persister.h"
#include "../../src/label.h"
#include "../../src/mesh_node.h"

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
  EXPECT_Vector3f_NEAR(vector, loaded, 1E-5f);
}

TEST(Test_QObjectPersistence, SaveAndLoadALabel)
{
  Label label(2, "my label 2", Eigen::Vector3f(1, 2, 3));

  Persister::save(label, "test.xml");
  auto loaded = Persister::load<Label>("test.xml");
  EXPECT_EQ(label.id, loaded.id);
  EXPECT_EQ(label.text, loaded.text);
  EXPECT_Vector3f_NEAR(label.anchorPosition, loaded.anchorPosition, 1E-5f);
}

TEST(Test_QObjectPersistence, SaveAndLoadALabelPointer)
{
  auto label = new Label(2, "my label 2", Eigen::Vector3f(1, 2, 3));

  Persister::save(label, "test.xml");
  auto loaded = Persister::load<Label*>("test.xml");
  EXPECT_EQ(label->id, loaded->id);
  EXPECT_EQ(label->text, loaded->text);
  EXPECT_Vector3f_NEAR(label->anchorPosition, loaded->anchorPosition, 1E-5f);

  delete label;
  delete loaded;
}

TEST(Test_QObjectPersistence, SaveAndLoadAMatrix)
{
  Eigen::Matrix4f matrix;
  matrix << 0.0f, 0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f,
      2.3f, 3.0f, 3.1f, 3.2f, 3.3f;

  Persister::save(matrix, "test.xml");
  auto loaded = Persister::load<Eigen::Matrix4f>("test.xml");
  EXPECT_Matrix4f_NEAR(matrix, loaded, 1E-5f);
}
