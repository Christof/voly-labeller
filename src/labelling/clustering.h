#ifndef SRC_LABELLING_CLUSTERING_H_

#define SRC_LABELLING_CLUSTERING_H_

#include <memory>
#include <map>
#include <random>
#include <vector>
#include "./label.h"
#include "../math/eigen.h"

class Labels;

/**
 * \brief Cluster given labels by their depth using k-means++
 *
 * The k-means++ algorithm is taken from:
 * Arthur, D., & Vassilvitskii, S. (n.d.).
 * k-means++: The Advantages of Careful Seeding, 8, 1–11.
 */
class Clustering
{
 public:
  Clustering(std::shared_ptr<Labels> labels, int clusterCount);

  void update(Eigen::Matrix4f viewProjectionMatrix);

  /** \brief Returns map of clusters (given by their z-value) to all their
   * labels (given as vector of label Ids).
   */
  std::map<float, std::vector<int>> getCentersWithLabelIds();

  /** \brief Returns map of clusters (given by farthest z-value of cluster
   * members) to all their labels (given as vector of label Ids).
   */
  std::map<float, std::vector<int>> getFarthestClusterMembersWithLabelIds();

  /** \brief Returns the median z-value for all clusters */
  std::vector<float> getMedianClusterMembers();
  std::map<float, std::vector<int>>
  getMedianClusterMembersWithLabelIdsInFront();

  int getRandomLabelIndexWeightedBy(std::vector<float> distances);

 private:
  std::shared_ptr<Labels> labels;
  int clusterCount;

  std::vector<Label> allLabels;

  std::vector<float> zValues;
  std::vector<int> clusterIndices;
  std::vector<float> clusterCenters;

  void initializeClusters();
  void recalculateZValues(Eigen::Matrix4f viewProjectionMatrix);
  std::vector<float>
  calculateDistancesToNearestCenter(size_t currentClusterCount);

  /** \brief Iterate over all labels, assign each label to the nearest clust
   * and recalculate the cluster centers
   *
   * \return Number of assignment changes
   */
  int updateStep();
  int findNearestCluster(float zValue, int clusterCount);
  int findNearestCluster(float zValue);
  void recalculateCenters();

  std::default_random_engine gen;
};

#endif  // SRC_LABELLING_CLUSTERING_H_
