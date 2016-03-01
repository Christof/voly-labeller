#ifndef SRC_LABELLING_CLUSTERING_H_

#define SRC_LABELLING_CLUSTERING_H_

#include <memory>
#include <map>
#include <vector>
#include "./label.h"
#include "../math/eigen.h"

class Labels;

/**
 * \brief
 *
 *
 */
class Clustering
{
 public:
  Clustering(std::shared_ptr<Labels> labels, int clusterCount);

  // returns map of clusters (given by their z-value) to all their labels
  // (given as vector of label Ids).
  std::map<float, std::vector<int>>
  update(Eigen::Matrix4f viewProjectionMatrix);

  // returns map of clusters (given by farthest z-value of clust members)
  // to all their labels // (given as vector of label Ids).
  std::map<float, std::vector<int>>
  updateAndReturnFarthestDepthValue(Eigen::Matrix4f viewProjectionMatrix);

 private:
  std::shared_ptr<Labels> labels;
  int clusterCount;

  std::map<int, std::vector<int>> clusterIndexToLabelIndices;

  std::vector<float> zValues;
  std::vector<int> clusterIndices;
  std::vector<float> clusterCenters;

  // returns number of changes
  int updateStep();
  int findNearestCluster(float zValue);
  void recalculateCenters();
};

#endif  // SRC_LABELLING_CLUSTERING_H_