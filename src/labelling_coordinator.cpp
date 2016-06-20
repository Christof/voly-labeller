#if _WIN32
#pragma warning(disable : 4267 4996)
#endif

#include "./labelling_coordinator.h"
#include <QtOpenGLExtensions>
#include <map>
#include <vector>
#include "./labelling/clustering.h"
#include "./labelling/labels.h"
#include "./placement/occlusion_calculator.h"
#include "./placement/constraint_updater.h"
#include "./placement/persistent_constraint_updater.h"
#include "./placement/cuda_texture_mapper.h"
#include "./placement/integral_costs_calculator.h"
#include "./placement/saliency.h"
#include "./placement/labels_arranger.h"
#include "./placement/insertion_order_labels_arranger.h"
#include "./placement/randomized_labels_arranger.h"
#include "./placement/apollonius_labels_arranger.h"
#include "./graphics/buffer_drawer.h"
#include "./nodes.h"
#include "./math/eigen.h"
#include "./label_node.h"
#include "./texture_mapper_manager.h"

LabellingCoordinator::LabellingCoordinator(
    int layerCount, std::shared_ptr<Forces::Labeller> forcesLabeller,
    std::shared_ptr<Labels> labels, std::shared_ptr<Nodes> nodes)
  : layerCount(layerCount), forcesLabeller(forcesLabeller), labels(labels),
    nodes(nodes), clustering(labels, layerCount - 1)
{
  occlusionCalculator =
      std::make_shared<Placement::OcclusionCalculator>(layerCount);
}

void LabellingCoordinator::initialize(
    int bufferSize, std::shared_ptr<Graphics::BufferDrawer> drawer,
    std::shared_ptr<TextureMapperManager> textureMapperManager, int width,
    int height)
{
  saliency = std::make_shared<Placement::Saliency>(
      textureMapperManager->getAccumulatedLayersTextureMapper(),
      textureMapperManager->getSaliencyTextureMapper());

  occlusionCalculator->initialize(textureMapperManager);
  integralCostsCalculator =
      std::make_shared<Placement::IntegralCostsCalculator>(
          textureMapperManager->getOcclusionTextureMapper(),
          textureMapperManager->getSaliencyTextureMapper(),
          textureMapperManager->getIntegralCostsTextureMapper());
  auto constraintUpdater =
      std::make_shared<ConstraintUpdater>(drawer, bufferSize, bufferSize);
  persistentConstraintUpdater =
      std::make_shared<PersistentConstraintUpdater>(constraintUpdater);

  insertionOrderLabelsArranger =
      std::make_shared<Placement::InsertionOrderLabelsArranger>();
  randomizedLabelsArranger =
      std::make_shared<Placement::RandomizedLabelsArranger>();

  for (int layerIndex = 0; layerIndex < layerCount; ++layerIndex)
  {
    auto labelsContainer = std::make_shared<LabelsContainer>();
    labelsInLayer.push_back(labelsContainer);
    auto labeller = std::make_shared<Placement::Labeller>(labelsContainer);
    labeller->resize(width, height);
    labeller->initialize(textureMapperManager->getIntegralCostsTextureMapper(),
                         textureMapperManager->getConstraintTextureMapper(),
                         persistentConstraintUpdater);

    auto apolloniusLabelsArranger =
        std::make_shared<Placement::ApolloniusLabelsArranger>();
    apolloniusLabelsArranger->initialize(
        textureMapperManager->getDistanceTransformTextureMapper(layerIndex),
        textureMapperManager->getOcclusionTextureMapper(),
        textureMapperManager->getApolloniusTextureMapper(layerIndex));
    apolloniusLabelsArrangers.push_back(apolloniusLabelsArranger);

    labeller->setLabelsArranger(useApollonius ? apolloniusLabelsArranger
                                              : insertionOrderLabelsArranger);

    placementLabellers.push_back(labeller);
  }
}

void LabellingCoordinator::cleanup()
{
  occlusionCalculator.reset();
  saliency.reset();
  integralCostsCalculator.reset();

  for (auto apolloniusLabelsArranger : apolloniusLabelsArrangers)
    apolloniusLabelsArranger->cleanup();

  for (auto placementLabeller : placementLabellers)
    placementLabeller->cleanup();
}

void LabellingCoordinator::update(double frameTime, Eigen::Matrix4f projection,
                                  Eigen::Matrix4f view, int activeLayerNumber)
{
  labellerFrameData = LabellerFrameData(frameTime, projection, view);

  saliency->runKernel();

  auto positionsNDC2d = getPlacementPositions(activeLayerNumber);
  auto positionsNDC = addDepthValueNDC(positionsNDC2d);

  std::map<int, Eigen::Vector3f> positions = ndcPositionsTo3d(positionsNDC);

  if (forcesEnabled)
    positions = getForcesPositions(positions);

  distributeLabelsToLayers();

  updateLabelPositionsInLabelNodes(positions);
}

void LabellingCoordinator::updatePlacement(bool isIdle)
{
  bool optimize = isIdle && optimizeOnIdle;
  float newSumOfCosts = 0.0f;
  persistentConstraintUpdater->clear();
  for (int layerIndex = 0; layerIndex < layerCount; ++layerIndex)
  {
    occlusionCalculator->calculateFor(layerIndex);
    integralCostsCalculator->runKernel();

    auto labeller = placementLabellers[layerIndex];
    auto defaultArranger = useApollonius ? apolloniusLabelsArrangers[layerIndex]
                                         : insertionOrderLabelsArranger;
    labeller->setLabelsArranger(optimize ? randomizedLabelsArranger
                                         : defaultArranger);
    labeller->update(labellerFrameData);
    newSumOfCosts += labeller->getLastSumOfCosts();
  }

  if (optimize)
  {
    std::cout << "Old costs: " << sumOfCosts << "\tnew costs:" << newSumOfCosts
              << std::endl;
  }

  if (optimize && newSumOfCosts > sumOfCosts)
  {
    preserveLastResult = true;
  }
  else
  {
    preserveLastResult = false;
    sumOfCosts = newSumOfCosts;
  }
}

std::vector<float> LabellingCoordinator::updateClusters()
{
  clustering.update(labellerFrameData.viewProjection);
  return clustering.getMedianClusterMembers();
  /*
  auto clusters = clustering.getFarthestClusterMembersWithLabelIds();
  std::vector<float> zValues;

  for (auto pair : clusters)
  {
    zValues.push_back(pair.first);
  }

  return zValues;
  */
}

void LabellingCoordinator::resize(int width, int height)
{
  for (auto placementLabeller : placementLabellers)
    placementLabeller->resize(width, height);

  forcesLabeller->resize(width, height);
}

void LabellingCoordinator::saveOcclusion()
{
  occlusionCalculator->saveOcclusion();
}

void LabellingCoordinator::setCostFunctionWeights(
    Placement::CostFunctionWeights weights)
{
  for (auto placementLabeller : placementLabellers)
    placementLabeller->setCostFunctionWeights(weights);
}

std::map<int, Eigen::Vector2f>
LabellingCoordinator::getPlacementPositions(int activeLayerNumber)
{
  if (preserveLastResult)
    return lastPlacementResult;

  std::map<int, Eigen::Vector2f> placementPositions;
  int layerIndex = 0;
  for (auto placementLabeller : placementLabellers)
  {
    if (activeLayerNumber == 0 || activeLayerNumber - 1 == layerIndex)
    {
      auto newPositionsForLayer = placementLabeller->getLastPlacementResult();
      placementPositions.insert(newPositionsForLayer.begin(),
                                newPositionsForLayer.end());
    }

    layerIndex++;
  }

  lastPlacementResult = placementPositions;

  return placementPositions;
}

std::map<int, Eigen::Vector3f> LabellingCoordinator::getForcesPositions(
    std::map<int, Eigen::Vector3f> placementPositions)
{
  if (firstFramesWithoutPlacement && placementPositions.size())
  {
    firstFramesWithoutPlacement = false;
    forcesLabeller->setPositions(labellerFrameData, placementPositions);
  }

  return forcesLabeller->update(labellerFrameData, placementPositions);
}

void LabellingCoordinator::distributeLabelsToLayers()
{
  auto centerWithLabelIds = clustering.getMedianClusterMembersWithLabelIds();
  int layerIndex = 0;
  for (auto &pair : centerWithLabelIds)
  {
    auto &container = labelsInLayer[layerIndex];
    container->clear();

    for (int labelId : pair.second)
    {
      container->add(labels->getById(labelId));
      labelIdToLayerIndex[labelId] = layerIndex;
      labelIdToZValue[labelId] = pair.first;
    }

    layerIndex++;
  }
}

void LabellingCoordinator::updateLabelPositionsInLabelNodes(
    std::map<int, Eigen::Vector3f> newPositions)
{
  for (auto &labelNode : nodes->getLabelNodes())
  {
    int labelId = labelNode->label.id;
    if (newPositions.count(labelId))
    {
      labelNode->setIsVisible(true);
      labelNode->labelPosition = newPositions[labelId];
      labelNode->layerIndex = labelIdToLayerIndex[labelId];
    }
    else
    {
      labelNode->setIsVisible(false);
    }
  }
}

std::map<int, Eigen::Vector3f> LabellingCoordinator::addDepthValueNDC(
    std::map<int, Eigen::Vector2f> positionsNDC)
{
  std::map<int, Eigen::Vector3f> positions;
  for (auto positionNDCPair : positionsNDC)
  {
    int labelId = positionNDCPair.first;
    auto position2d = positionNDCPair.second;
    positions[labelId] = Eigen::Vector3f(position2d.x(), position2d.y(),
                                         labelIdToZValue[labelId]);
  }

  return positions;
}

std::map<int, Eigen::Vector3f> LabellingCoordinator::ndcPositionsTo3d(
    std::map<int, Eigen::Vector3f> positionsNDC)
{
  Eigen::Matrix4f inverseViewProjection =
      labellerFrameData.viewProjection.inverse();

  std::map<int, Eigen::Vector3f> positions;
  for (auto positionNDCPair : positionsNDC)
  {
    int labelId = positionNDCPair.first;
    positions[labelId] = project(inverseViewProjection, positionNDCPair.second);
  }

  return positions;
}

