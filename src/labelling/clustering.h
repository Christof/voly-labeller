#ifndef SRC_LABELLING_CLUSTERING_H_

#define SRC_LABELLING_CLUSTERING_H_

#include <memory>
#include <map>
#include <vector>
#include "./label.h"
#include "../math/eigen.h"

class Labels;

/**
 * \brief Cluster given labels by their depth using k-means
 *
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

  /** \brief Returns map of clusters (given by farthest z-value of clust
   * members)
   * to all their labels // (given as vector of label Ids).
   */
  std::map<float, std::vector<int>> getFarthestClusterMembersWithLabelIds();

 private:
  std::shared_ptr<Labels> labels;
  int clusterCount;

  std::vector<Label> allLabels;

  std::vector<float> zValues;
  std::vector<int> clusterIndices;
  std::vector<float> clusterCenters;

  /** \brief Iterate over all labels, assign each label to the nearest clust
   * and recalculate the cluster centers
   *
   * \return Number of assignment changes
   */
  int updateStep();
  int findNearestCluster(float zValue);
  void recalculateCenters();
};

#endif  // SRC_LABELLING_CLUSTERING_H_
