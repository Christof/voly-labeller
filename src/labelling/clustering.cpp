#include "./clustering.h"
#include <map>
#include <random>
#include <vector>
#include <limits>
#include "./labels.h"

Clustering::Clustering(std::shared_ptr<Labels> labels, int clusterCount)
  : labels(labels), clusterCount(clusterCount), clusterCenters(clusterCount)
{
}

void Clustering::update(Eigen::Matrix4f viewProjectionMatrix)
{
  if (labels->count() == 0)
    return;

  recalculateZValues(viewProjectionMatrix);

  initializeClusters();

  recalculateCenters();

  int updateCount = 0;
  do
  {
    updateCount = updateStep();
  } while (updateCount != 0);
}

std::map<float, std::vector<int>> Clustering::getCentersWithLabelIds()
{
  int labelIndex = 0;
  std::map<float, std::vector<int>> result;
  for (auto &label : allLabels)
  {
    int clusterIndex = clusterIndices[labelIndex];
    result[clusterCenters[clusterIndex]].push_back(label.id);
    ++labelIndex;
  }

  return result;
}

std::map<float, std::vector<int>>
Clustering::getFarthestClusterMembersWithLabelIds()
{
  std::map<float, std::vector<int>> result;
  std::map<int, std::vector<float>> clusterIndexToZValues;
  std::map<int, std::vector<int>> clusterIndexToLabelIds;
  int labelIndex = 0;
  for (auto &label : allLabels)
  {
    int clusterIndex = clusterIndices[labelIndex];
    clusterIndexToZValues[clusterIndex].push_back(zValues[labelIndex]);
    clusterIndexToLabelIds[clusterIndex].push_back(label.id);
    ++labelIndex;
  }

  for (auto &pair : clusterIndexToZValues)
  {
    float maxDepth = *std::max_element(pair.second.begin(), pair.second.end());

    result[maxDepth] = clusterIndexToLabelIds[pair.first];
  }

  return result;
}

float median(std::vector<float> &vector)
{
  size_t halfSize = vector.size() / 2;
  std::nth_element(vector.begin(), vector.begin() + halfSize, vector.end());
  return vector[halfSize];
}

std::vector<float> Clustering::getMedianClusterMembers()
{
  std::map<int, std::vector<float>> clusterIndexToZValues;
  for (size_t labelIndex = 0; labelIndex < allLabels.size(); ++labelIndex)
  {
    int clusterIndex = clusterIndices[labelIndex];
    clusterIndexToZValues[clusterIndex].push_back(zValues[labelIndex]);
  }

  std::vector<float> medians;
  for (auto &pair : clusterIndexToZValues)
  {
    medians.push_back(median(pair.second));
  }

  return medians;
}

float getDistance(float value1, float value2)
{
  float diff = value1 - value2;

  return diff * diff;
}

int Clustering::getRandomLabelIndexWeightedBy(std::vector<float> distances)
{
  float sumOfDistances =
      std::accumulate(distances.begin(), distances.end(), 0.0f);

  std::uniform_real_distribution<float> dist(0.0f, sumOfDistances);
  float randomValue = dist(gen);

  float sum = 0.0f;
  for (size_t labelIndex = 0; labelIndex < distances.size(); ++labelIndex)
  {
    sum += distances[labelIndex];

    if (sum >= randomValue)
      return labelIndex;
  }

  return distances.size() - 1;
}

void Clustering::recalculateZValues(Eigen::Matrix4f viewProjectionMatrix)
{
  allLabels = labels->getLabels();

  zValues.clear();
  int labelIndex = 0;
  for (auto &label : allLabels)
  {
    zValues.push_back(project(viewProjectionMatrix, label.anchorPosition).z());
    ++labelIndex;
  }
}

void Clustering::initializeClusters()
{
  std::vector<int> labelIndexForClusterIndex;
  gen.seed(0);

  std::uniform_int_distribution<> dist(0, zValues.size() - 1);
  int clusterIndex = dist(gen);
  clusterCenters[0] = zValues[clusterIndex];
  labelIndexForClusterIndex.push_back(clusterIndex);

  for (size_t clusterIndex = 1; clusterIndex < clusterCenters.size();
       ++clusterIndex)
  {
    auto distances = calculateDistancesToNearestCenter(clusterIndex);

    int newClusterIndex = getRandomLabelIndexWeightedBy(distances);
    clusterCenters[clusterIndex] = zValues[newClusterIndex];
    labelIndexForClusterIndex.push_back(newClusterIndex);
  }

  for (size_t labelIndex = 0; labelIndex < zValues.size(); ++labelIndex)
  {
    clusterIndices.push_back(findNearestCluster(zValues[labelIndex]));
  }
}

std::vector<float>
Clustering::calculateDistancesToNearestCenter(size_t currentClusterCount)
{
  std::vector<float> distances;
  for (size_t labelIndex = 0; labelIndex < zValues.size(); ++labelIndex)
  {
    float zValue = zValues[labelIndex];
    int nearestCluster = findNearestCluster(zValue, currentClusterCount);
    distances.push_back(getDistance(clusterCenters[nearestCluster], zValue));
  }

  return distances;
}

int Clustering::updateStep()
{
  int updateCount = 0;

  for (size_t labelIndex = 0; labelIndex < zValues.size(); ++labelIndex)
  {
    float zValue = zValues[labelIndex];
    int newClusterIndex = findNearestCluster(zValue);

    if (clusterIndices[labelIndex] != newClusterIndex)
    {
      ++updateCount;
      clusterIndices[labelIndex] = newClusterIndex;
    }
  }

  if (updateCount > 0)
    recalculateCenters();

  return updateCount;
}

int Clustering::findNearestCluster(float zValue, int clusterCount)
{
  float minDistance = std::numeric_limits<float>::max();
  int bestClusterIndex = -1;
  for (int clusterIndex = 0; clusterIndex < clusterCount; ++clusterIndex)
  {
    float distance = getDistance(clusterCenters[clusterIndex], zValue);

    if (distance < minDistance)
    {
      minDistance = distance;
      bestClusterIndex = clusterIndex;
    }
  }

  return bestClusterIndex;
}

int Clustering::findNearestCluster(float zValue)
{
  return findNearestCluster(zValue, clusterCount);
}

void Clustering::recalculateCenters()
{
  std::vector<float> sums(clusterCount, 0.0f);
  std::vector<int> counter(clusterCount, 0);

  for (size_t labelIndex = 0; labelIndex < zValues.size(); ++labelIndex)
  {
    int clusterIndex = clusterIndices[labelIndex];
    sums[clusterIndex] += zValues[labelIndex];
    ++counter[clusterIndex];
  }

  for (int clusterIndex = 0; clusterIndex < clusterCount; ++clusterIndex)
  {
    clusterCenters[clusterIndex] = sums[clusterIndex] / counter[clusterIndex];
  }
}

