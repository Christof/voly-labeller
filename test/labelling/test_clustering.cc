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

TEST(Test_Clustering, ForClusterCountOf0)
{
  auto labels = std::make_shared<Labels>();
  labels->add(Label(1, "Label 1", Eigen::Vector3f(0, 0, 1)));
  labels->add(Label(2, "Label 2", Eigen::Vector3f(0, 0, 1)));

  Clustering clustering(labels, 0);

  clustering.update(Eigen::Matrix4f::Identity());
  auto centersWithLabelIds = clustering.getCentersWithLabelIds();
  EXPECT_EQ(1, centersWithLabelIds.size());
  EXPECT_EQ(2, centersWithLabelIds[1.0f].size());
  EXPECT_EQ(1, centersWithLabelIds[1.0f][0]);
  EXPECT_EQ(2, centersWithLabelIds[1.0f][1]);

  EXPECT_EQ(1, clustering.getMedianClusterMembers().size());
  EXPECT_EQ(0, clustering.getFarthestClusterMembersWithLabelIds().size());

  auto medianWithLabelIdsInFront = clustering.getMedianClusterMembersWithLabelIdsInFront();
  EXPECT_EQ(1, medianWithLabelIdsInFront.size());
  EXPECT_EQ(2, medianWithLabelIdsInFront[1.0f].size());
  EXPECT_EQ(1, medianWithLabelIdsInFront[1.0f][0]);
  EXPECT_EQ(2, medianWithLabelIdsInFront[1.0f][1]);
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

TEST(Test_Clustering, UpdateAndReturnFarthestZValueForMoreLabelsThanClusters)
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

TEST(Test_Clustering, UpdateAndCheckMedianMembers)
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
  auto result = clustering.getMedianClusterMembers();

  ASSERT_EQ(3, result.size());

  EXPECT_FLOAT_EQ(0.9f, result[0]);
  EXPECT_FLOAT_EQ(0.35f, result[1]);
  EXPECT_FLOAT_EQ(0.1f, result[2]);
}

TEST(Test_Clustering, UpdateAndCheckClustersCreatedByMedian)
{
  auto labels = std::make_shared<Labels>();

  labels->add(Label(1, "Label 0", Eigen::Vector3f(0, 0, 1)));
  labels->add(Label(2, "Label 1", Eigen::Vector3f(0, 0, 0.7)));
  labels->add(Label(4, "Label 2", Eigen::Vector3f(0, 0, 0.4)));
  labels->add(Label(6, "Label 3", Eigen::Vector3f(0, 0, 0)));
  labels->add(Label(8, "Label 4", Eigen::Vector3f(0, 0, 0.3)));
  labels->add(Label(10, "Label 5", Eigen::Vector3f(0, 0, 0.35)));
  labels->add(Label(12, "Label 6", Eigen::Vector3f(0, 0, 0.1)));
  labels->add(Label(14, "Label 7", Eigen::Vector3f(0, 0, 0.9)));

  Clustering clustering(labels, 3);

  clustering.update(Eigen::Matrix4f::Identity());
  auto result = clustering.getMedianClusterMembersWithLabelIdsInFront();

  ASSERT_EQ(4, result.size());

  auto pairIterator = result.begin();
  auto indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.1, pairIterator->first);
  EXPECT_EQ(2, indices.size());
  EXPECT_EQ(6, indices[0]);
  EXPECT_EQ(12, indices[1]);

  ++pairIterator;
  indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.35, pairIterator->first);
  EXPECT_EQ(2, indices.size());
  EXPECT_EQ(8, indices[0]);
  EXPECT_EQ(10, indices[1]);

  ++pairIterator;
  indices = pairIterator->second;
  EXPECT_FLOAT_EQ(0.9, pairIterator->first);
  EXPECT_EQ(3, indices.size());
  EXPECT_EQ(2, indices[0]);
  EXPECT_EQ(4, indices[1]);
  EXPECT_EQ(14, indices[2]);

  ++pairIterator;
  indices = pairIterator->second;
  EXPECT_FLOAT_EQ(1.0, pairIterator->first);
  EXPECT_EQ(1, indices.size());
  EXPECT_EQ(1, indices[0]);
}

TEST(Test_Clustering, getRandomLabelIndexWeightedBy_equal_distances)
{
  auto labels = std::make_shared<Labels>();
  Clustering clustering(labels, 3);

  std::vector<float> equalDistances = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

  std::vector<float> winnerCountEqual(equalDistances.size(), 0);

  for (int i = 0; i < 10000; ++i)
  {
    int index = clustering.getRandomLabelIndexWeightedBy(equalDistances);
    winnerCountEqual[index] += 0.0001;
  }

  EXPECT_NEAR(0.2f, winnerCountEqual[0], 0.03f);
  EXPECT_NEAR(0.2f, winnerCountEqual[1], 0.03f);
  EXPECT_NEAR(0.2f, winnerCountEqual[2], 0.03f);
  EXPECT_NEAR(0.2f, winnerCountEqual[3], 0.03f);
  EXPECT_NEAR(0.2f, winnerCountEqual[4], 0.03f);
}

TEST(Test_Clustering, getRandomLabelIndexWeightedBy_different_distances)
{
  auto labels = std::make_shared<Labels>();
  Clustering clustering(labels, 3);
  std::vector<float> differentDistances = { 1.0f, 3.0f, 2.0f, 4.0f, 0.0f };
  std::vector<float> winnerCountDifferent(differentDistances.size(), 0);

  for (int i = 0; i < 10000; ++i)
  {
    int index = clustering.getRandomLabelIndexWeightedBy(differentDistances);
    winnerCountDifferent[index] += 0.0001;
  }

  EXPECT_NEAR(0.1f, winnerCountDifferent[0], 0.03f);
  EXPECT_NEAR(0.3f, winnerCountDifferent[1], 0.03f);
  EXPECT_NEAR(0.2f, winnerCountDifferent[2], 0.03f);
  EXPECT_NEAR(0.4f, winnerCountDifferent[3], 0.03f);
  EXPECT_EQ(0.0f, winnerCountDifferent[4]);
}
