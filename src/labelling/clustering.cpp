#include "./clustering.h"
#include <map>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>
#include "./labels.h"

Clustering::Clustering(std::shared_ptr<Labels> labels, int clusterCount)
  : labels(labels), clusterCount(clusterCount), clusterCenters(clusterCount)
{
}

void Clustering::update(Eigen::Matrix4f viewProjectionMatrix)
{
  recalculateZValues(viewProjectionMatrix);

  if (allLabels.size() == 0)
    return;

  if (clusterCount == 0)
    return;

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
  if (clusterCount == 0)
  {
    std::map<float, std::vector<int>> result;
    for (auto &label : allLabels)
      result[1.0].push_back(label.id);

    return result;
  }

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
  if (clusterCount == 0)
    return std::map<float, std::vector<int>>();

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
  if (clusterCount == 0)
    return std::vector<float>{ 1.0f };

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

std::map<float, std::vector<int>>
Clustering::getMedianClusterMembersWithLabelIdsInFront()
{
  std::vector<float> medians = getMedianClusterMembers();
  std::sort(medians.begin(), medians.end());

  std::map<float, std::vector<int>> result;
  std::vector<int> labelIndicesToPlace;
  for (size_t labelIndex = 0; labelIndex < allLabels.size(); ++labelIndex)
    labelIndicesToPlace.push_back(labelIndex);

  for (float median : medians)
  {
    for (auto iterator = labelIndicesToPlace.cbegin();
         iterator != labelIndicesToPlace.cend();)
    {
      auto labelIndex = *iterator;
      if (zValues[labelIndex] <= median)
      {
        result[median].push_back(allLabels[labelIndex].id);
        iterator = labelIndicesToPlace.erase(iterator);
      }
      else
      {
        ++iterator;
      }
    }
  }

  for (auto labelIndex : labelIndicesToPlace)
    result[1].push_back(allLabels[labelIndex].id);

  return result;
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
  allLabels.clear();
  zValues.clear();
  for (auto label : labels->getLabels())
  {
    Eigen::Vector3f anchorPositionNDC =
        project(viewProjectionMatrix, label.anchorPosition);

    if (!label.isAnchorInsideFieldOfView(viewProjectionMatrix))
      continue;

    allLabels.push_back(label);
    zValues.push_back(anchorPositionNDC.z());
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

