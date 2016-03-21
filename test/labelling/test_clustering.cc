#include "../test.h"
#include "../../src/labelling/clustering.h"
#include "../../src/labelling/labels.h"

TEST(Test_Clustering, ForNoLabel)
{
  auto labels = std::make_shared<Labels>();

  Clustering clustering(labels, 4);

  clustering.update(Eigen::Matrix4f::Identity());
  auto result = clustering.getCentersWithLabelIds();

  EXPECT_EQ(0, result.size());
}

TEST(Test_Clustering, ForAsManyLabelsAsClusters)
{
  auto labels = std::make_shared<Labels>();

  labels->add(Label(0, "Label 0", Eigen::Vector3f(0, 0, 1)));
  labels->add(Label(1, "Label 1", Eigen::Vector3f(0, 0, 0.7)));
  labels->add(Label(2, "Label 2", Eigen::Vector3f(0, 0, 0.4)));
  labels->add(Label(3, "Label 3", Eigen::Vector3f(0, 0, 0)));

  Clustering clustering(labels, 4);

  clustering.update(Eigen::Matrix4f::Identity());
  auto result = clustering.getCentersWithLabelIds();

  ASSERT_EQ(4, result.size());

  EXPECT_EQ(1, result[0.0f].size());
  EXPECT_EQ(3, result[0.0f][0]);

  EXPECT_EQ(1, result[0.4f].size());
  EXPECT_EQ(2, result[0.4f][0]);

  EXPECT_EQ(1, result[0.7f].size());
  EXPECT_EQ(1, result[0.7f][0]);

  EXPECT_EQ(1, result[1.0f].size());
  EXPECT_EQ(0, result[1.0f][0]);
}

TEST(Test_Clustering, ForAsMoreLabelsThanClusters)
{
  auto labels = std::make_shared<Labels>();

  labels->add(Label(0, "Label 0", Eigen::Vector3f(0, 0, 1)));
  labels->add(Label(1, "Label 1", Eigen::Vector3f(0, 0, 0.7)));
  labels->add(Label(2, "Label 2", Eigen::Vector3f(0, 0, 0.4)));
  labels->add(Label(3, "Label 3", Eigen::Vector3f(0, 0, 0)));
  labels->add(Label(4, "Label 4", Eigen::Vector3f(0, 0, 0.3)));
  labels->add(Label(5, "Label 5", Eigen::Vector3f(0, 0, 0.35)));
  labels->add(Label(6, "Label 6", Eigen::Vector3f(0, 0, 0.1)));
  labels->add(Label(7, "Label 7", Eigen::Vector3f(0, 0, 0.9)));

  Clustering clustering(labels, 3);

  clustering.update(Eigen::Matrix4f::Identity());
  auto result = clustering.getCentersWithLabelIds();

  ASSERT_EQ(3, result.size());

  auto pairIterator = result.begin();
  auto indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.05, pairIterator->first);
  EXPECT_EQ(2, indices.size());
  EXPECT_EQ(3, indices[0]);
  EXPECT_EQ(6, indices[1]);

  ++pairIterator;
  indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.35, pairIterator->first);
  EXPECT_EQ(3, indices.size());
  EXPECT_EQ(2, indices[0]);
  EXPECT_EQ(4, indices[1]);
  EXPECT_EQ(5, indices[2]);

  ++pairIterator;
  indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.8666666, pairIterator->first);
  EXPECT_EQ(3, indices.size());
  EXPECT_EQ(0, indices[0]);
  EXPECT_EQ(1, indices[1]);
  EXPECT_EQ(7, indices[2]);
}

TEST(Test_Clustering, ForAsMoreLabelsThanClustersWhereLabelsAreAtTheFarEnd)
{
  auto labels = std::make_shared<Labels>();

  labels->add(Label(0, "Label 0", Eigen::Vector3f(0, 0, 0.8)));
  labels->add(Label(1, "Label 1", Eigen::Vector3f(0, 0, 0.7)));
  labels->add(Label(2, "Label 2", Eigen::Vector3f(0, 0, 0.9)));
  labels->add(Label(3, "Label 3", Eigen::Vector3f(0, 0, 0.95)));
  labels->add(Label(4, "Label 4", Eigen::Vector3f(0, 0, 0.87)));
  labels->add(Label(5, "Label 5", Eigen::Vector3f(0, 0, 0.95)));
  labels->add(Label(6, "Label 6", Eigen::Vector3f(0, 0, 0.89)));
  labels->add(Label(7, "Label 7", Eigen::Vector3f(0, 0, 0.9)));

  Clustering clustering(labels, 3);

  clustering.update(Eigen::Matrix4f::Identity());
  auto result = clustering.getCentersWithLabelIds();

  ASSERT_EQ(3, result.size());

  auto pairIterator = result.begin();
  auto indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.75, pairIterator->first);
  EXPECT_EQ(2, indices.size());
  EXPECT_EQ(0, indices[0]);
  EXPECT_EQ(1, indices[1]);

  ++pairIterator;
  indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.89, pairIterator->first);
  EXPECT_EQ(4, indices.size());
  EXPECT_EQ(2, indices[0]);
  EXPECT_EQ(4, indices[1]);
  EXPECT_EQ(6, indices[2]);
  EXPECT_EQ(7, indices[3]);

  ++pairIterator;
  indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.95, pairIterator->first);
  EXPECT_EQ(2, indices.size());
  EXPECT_EQ(3, indices[0]);
  EXPECT_EQ(5, indices[1]);
}

TEST(Test_Clustering, UpdateAndReturnFarthestZValueForAsMoreLabelsThanClusters)
{
  auto labels = std::make_shared<Labels>();

  labels->add(Label(0, "Label 0", Eigen::Vector3f(0, 0, 1)));
  labels->add(Label(1, "Label 1", Eigen::Vector3f(0, 0, 0.7)));
  labels->add(Label(2, "Label 2", Eigen::Vector3f(0, 0, 0.4)));
  labels->add(Label(3, "Label 3", Eigen::Vector3f(0, 0, 0)));
  labels->add(Label(4, "Label 4", Eigen::Vector3f(0, 0, 0.3)));
  labels->add(Label(5, "Label 5", Eigen::Vector3f(0, 0, 0.35)));
  labels->add(Label(6, "Label 6", Eigen::Vector3f(0, 0, 0.1)));
  labels->add(Label(7, "Label 7", Eigen::Vector3f(0, 0, 0.9)));

  Clustering clustering(labels, 3);

  clustering.update(Eigen::Matrix4f::Identity());
  auto result = clustering.getFarthestClusterMembersWithLabelIds();

  ASSERT_EQ(3, result.size());

  auto pairIterator = result.begin();
  auto indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.1, pairIterator->first);
  EXPECT_EQ(2, indices.size());
  EXPECT_EQ(3, indices[0]);
  EXPECT_EQ(6, indices[1]);

  ++pairIterator;
  indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.4, pairIterator->first);
  EXPECT_EQ(3, indices.size());
  EXPECT_EQ(2, indices[0]);
  EXPECT_EQ(4, indices[1]);
  EXPECT_EQ(5, indices[2]);

  ++pairIterator;
  indices = pairIterator->second;
  EXPECT_FLOAT_EQ(1.0, pairIterator->first);
  EXPECT_EQ(3, indices.size());
  EXPECT_EQ(0, indices[0]);
  EXPECT_EQ(1, indices[1]);
  EXPECT_EQ(7, indices[2]);
}

