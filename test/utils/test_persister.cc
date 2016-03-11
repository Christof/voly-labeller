#include "../test.h"
#include <Eigen/Core>
#include "../../src/utils/persister.h"
#include "../../src/labelling/label.h"
#include "../../src/mesh_node.h"
#include "../../src/label_node.h"

TEST(Test_Persister, SaveAndLoadASimpleString)
{
  std::string text = "example text";

  Persister::save(text, "test.xml");

  EXPECT_EQ(text, Persister::load<std::string>("test.xml"));
}

TEST(Test_Persister, SaveAndLoadAVector3f)
{
  Eigen::Vector3f vector(1.0f, 1.1f, 1.2f);

  Persister::save(vector, "test.xml");
  auto loaded = Persister::load<Eigen::Vector3f>("test.xml");
  EXPECT_Vector3f_NEAR(vector, loaded, 1E-5f);
}

TEST(Test_Persister, SaveAndLoadAVector2f)
{
  Eigen::Vector2f vector(1.0f, 1.1f);

  Persister::save(vector, "test.xml");
  auto loaded = Persister::load<Eigen::Vector2f>("test.xml");
  EXPECT_Vector2f_NEAR(vector, loaded, 1E-5f);
}

TEST(Test_Persister, SaveAndLoadALabel)
{
  Label label(2, "my label 2", Eigen::Vector3f(1, 2, 3));

  Persister::save(label, "test.xml");
  auto loaded = Persister::load<Label>("test.xml");
  EXPECT_EQ(label.id, loaded.id);
  EXPECT_EQ(label.text, loaded.text);
  EXPECT_Vector3f_NEAR(label.anchorPosition, loaded.anchorPosition, 1E-5f);
}

TEST(Test_Persister, SaveAndLoadALabelPointer)
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

TEST(Test_Persister, SaveAndLoadAVectorOfLabelPointers)
{
  auto label = new Label(1, "my label 1", Eigen::Vector3f(1, 2, 3));
  auto label2 = new Label(2, "my label 2", Eigen::Vector3f(4, 5, 6));

  std::vector<Label*> labels;
  labels.push_back(label);
  labels.push_back(label2);

  Persister::save(labels, "test.xml");
  auto loaded = Persister::load<std::vector<Label*>>("test.xml");
  auto loadedLabel1 = loaded[0];
  auto loadedLabel2 = loaded[1];
  EXPECT_EQ(label->id, loadedLabel1->id);
  EXPECT_EQ(label->text, loadedLabel1->text);
  EXPECT_Vector3f_NEAR(label->anchorPosition, loadedLabel1->anchorPosition, 1E-5f);

  EXPECT_EQ(label2->id, loadedLabel2->id);
  EXPECT_EQ(label2->text, loadedLabel2->text);
  EXPECT_Vector3f_NEAR(label2->anchorPosition, loadedLabel2->anchorPosition, 1E-5f);

  delete label;
  delete label2;
  delete loaded[0];
  delete loaded[1];
}

TEST(Test_Persister, SaveAndLoadAMatrix)
{
  Eigen::Matrix4f matrix;
  matrix << 0.0f, 0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f,
      2.3f, 3.0f, 3.1f, 3.2f, 3.3f;

  Persister::save(matrix, "test.xml");
  auto loaded = Persister::load<Eigen::Matrix4f>("test.xml");
  EXPECT_Matrix4f_NEAR(matrix, loaded, 1E-5f);
}
