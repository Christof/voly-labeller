#if _WIN32
#pragma warning(disable : 4267 4996)
#endif

#include "./labelling_coordinator.h"
#include <QtOpenGLExtensions>
#include <QLoggingCategory>
#include <map>
#include <vector>
#include <memory>
#include "./labelling/clustering.h"
#include "./labelling/labels.h"
#include "./placement/occlusion_calculator.h"
#include "./placement/constraint_updater.h"
#include "./placement/persistent_constraint_updater.h"
#include "./placement/cuda_texture_mapper.h"
#include "./placement/cuda_texture_3d_mapper.h"
#include "./placement/integral_costs_calculator.h"
#include "./placement/direct_integral_costs_calculator.h"
#include "./placement/saliency.h"
#include "./placement/labels_arranger.h"
#include "./placement/insertion_order_labels_arranger.h"
#include "./placement/randomized_labels_arranger.h"
#include "./placement/apollonius_labels_arranger.h"
#include "./placement/anchor_constraint_drawer.h"
#include "./placement/shadow_constraint_drawer.h"
#include "./graphics/buffer_drawer.h"
#include "./nodes.h"
#include "./math/eigen.h"
#include "./label_node.h"
#include "./texture_mapper_manager.h"
#include "./utils/profiler.h"
#include "./utils/profiling_statistics.h"
#include "./graphics/managers.h"

QLoggingCategory lcChan("LabellingCoordinator");

LabellingCoordinator::LabellingCoordinator(
    int layerCount, std::shared_ptr<Forces::Labeller> forcesLabeller,
    std::shared_ptr<Labels> labels, std::shared_ptr<Nodes> nodes)
  : layerCount(layerCount), forcesLabeller(forcesLabeller), labels(labels),
    nodes(nodes), clustering(labels, layerCount - 1),
    profilingStatistics("LabellingCoordinator", lcChan)
{
  occlusionCalculator =
      std::make_shared<Placement::OcclusionCalculator>(layerCount);
}

void LabellingCoordinator::initialize(
    Graphics::Gl *gl, int bufferSize,
    std::shared_ptr<Graphics::Managers> managers,
    std::shared_ptr<TextureMapperManager> textureMapperManager, int width,
    int height)
{
  qCInfo(lcChan) << "Initialize";
  this->bufferSize = Eigen::Vector2f(bufferSize, bufferSize);
  saliency = std::make_shared<Placement::Saliency>(
      textureMapperManager->getAccumulatedLayersTextureMapper(),
      textureMapperManager->getSaliencyTextureMapper());

  occlusionCalculator->initialize(textureMapperManager);

  directIntegralCostsCalculator =
      std::make_shared<Placement::DirectIntegralCostsCalculator>(
          textureMapperManager->getColorTextureMapper(),
          textureMapperManager->getSaliencyTextureMapper(),
          textureMapperManager->getIntegralCostsTextureMapper());

  auto shaderManager = managers->getShaderManager();
  auto anchorConstraintDrawer =
      std::make_shared<AnchorConstraintDrawer>(bufferSize, bufferSize);
  anchorConstraintDrawer->initialize(gl, shaderManager);

  auto connectorShadowDrawer =
      std::make_shared<ShadowConstraintDrawer>(bufferSize, bufferSize);
  connectorShadowDrawer->initialize(gl, shaderManager);

  auto shadowConstraintDrawer =
      std::make_shared<ShadowConstraintDrawer>(bufferSize, bufferSize);
  shadowConstraintDrawer->initialize(gl, shaderManager);

  float scaleFactor = bufferSize / width;
  auto constraintUpdater = std::make_shared<ConstraintUpdater>(
      bufferSize, bufferSize, anchorConstraintDrawer, connectorShadowDrawer,
      shadowConstraintDrawer, scaleFactor);
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
  directIntegralCostsCalculator.reset();

  for (auto apolloniusLabelsArranger : apolloniusLabelsArrangers)
    apolloniusLabelsArranger->cleanup();

  for (auto placementLabeller : placementLabellers)
    placementLabeller->cleanup();
}

void LabellingCoordinator::setEnabled(bool enabled)
{
  labellingEnabled = enabled;
  for (auto &labelNode : nodes->getLabelNodes())
  {
    labelNode->setIsVisible(!enabled);
  }
}

bool LabellingCoordinator::update(double frameTime, bool isIdle,
                                  Eigen::Matrix4f projection,
                                  Eigen::Matrix4f view, int activeLayerNumber)
{
  Profiler profiler("update", lcChan, &profilingStatistics);

  if (!labellingEnabled)
    return false;

  this->isIdle = isIdle;
  labellerFrameData = LabellerFrameData(frameTime, projection, view);

  if (internalLabellingEnabled)
  {
    performInternalLabelling();
    return false;
  }

  saliency->runKernel();

  auto positionsNDC2d = getPlacementPositions(activeLayerNumber);
  auto positionsNDC = addDepthValueNDC(positionsNDC2d);
  LabelPositions labelPositions(positionsNDC, ndcPositionsTo3d(positionsNDC));

  if (forcesEnabled)
    labelPositions = getForcesPositions(labelPositions);

  distributeLabelsToLayers();

  updateLabelPositionsInLabelNodes(labelPositions);

  return hasChanges;
}

void LabellingCoordinator::updatePlacement()
{
  Profiler profiler("updatePlacement", lcChan, &profilingStatistics);

  if (!labellingEnabled)
    return;

  bool optimize = isIdle && optimizeOnIdle;
  bool ignoreOldPosition = !isIdle;

  float newSumOfCosts = 0.0f;
  persistentConstraintUpdater->clear();
  if (saveConstraintsInNextFrame)
  {
    persistentConstraintUpdater->save();
    saveConstraintsInNextFrame = false;
  }

  std::vector<Eigen::Vector2f> anchorPositions(labels->count());
  auto labelVector = labels->getLabels();
  for (int index = 0; index < labels->count(); index++)
  {
    auto anchor2D =
        labellerFrameData.project(labelVector[index].anchorPosition);
    anchorPositions[index] = toPixel(anchor2D.head<2>(), bufferSize);
  }

  persistentConstraintUpdater->setAnchorPositions(anchorPositions);

  for (int layerIndex = 0; layerIndex < layerCount; ++layerIndex)
  {
    if (useApollonius)
      occlusionCalculator->calculateFor(layerIndex);

    directIntegralCostsCalculator->runKernel(layerIndex, layerCount);

    auto labeller = placementLabellers[layerIndex];
    auto defaultArranger = useApollonius ? apolloniusLabelsArrangers[layerIndex]
                                         : insertionOrderLabelsArranger;
    labeller->setLabelsArranger(optimize ? randomizedLabelsArranger
                                         : defaultArranger);
    labeller->update(labellerFrameData, ignoreOldPosition);
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
  if (!labellingEnabled || internalLabellingEnabled)
    return std::vector<float>{ 1.0f };

  clustering.update(labellerFrameData.viewProjection);
  return clustering.getMedianClusterMembers();
}

bool LabellingCoordinator::haveLabelPositionsChanged()
{
  return labellingEnabled && hasChanges;
}

void LabellingCoordinator::resize(int width, int height)
{
  for (auto placementLabeller : placementLabellers)
    placementLabeller->resize(width, height);

  forcesLabeller->resize(width, height);
}

void LabellingCoordinator::toggleAnchorVisibility()
{
  for (auto node : nodes->getLabelNodes())
    node->isAnchorVisible = !node->isAnchorVisible;
}

void LabellingCoordinator::saveOcclusion()
{
  occlusionCalculator->saveOcclusion();
}

void LabellingCoordinator::saveConstraints()
{
  saveConstraintsInNextFrame = true;
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

LabelPositions
LabellingCoordinator::getForcesPositions(LabelPositions placementPositions)
{
  if (placementPositions.size() == 0)
    return LabelPositions();

  if (firstFramesWithoutPlacement && placementPositions.size())
  {
    firstFramesWithoutPlacement = false;
    forcesLabeller->overallForceFactor = isIdle ? 6.0f : 3.0f;
    forcesLabeller->setPositions(labellerFrameData, placementPositions);
  }

  return forcesLabeller->update(labellerFrameData, placementPositions);
}

void LabellingCoordinator::distributeLabelsToLayers()
{
  auto centerWithLabelIds = clustering.getMedianClusterMembersWithLabelIds();
  labelIdToLayerIndex.clear();
  labelIdToZValue.clear();

  for (auto &layerLabels : labelsInLayer)
    layerLabels->clear();

  int layerIndex = 0;
  for (auto &pair : centerWithLabelIds)
  {
    auto &container = labelsInLayer[layerIndex];

    for (int labelId : pair.second)
    {
      container->add(labels->getById(labelId));
      labelIdToLayerIndex[labelId] = layerIndex;
      labelIdToZValue[labelId] = pair.first;
    }

    layerIndex++;
  }

  if (lcChan.isDebugEnabled())
  {
    std::stringstream output;
    int layerIndex = 0;
    for (auto layerLabels : labelsInLayer)
    {
      output << std::endl << "Layer " << layerIndex++ << "\t";
      for (auto &label : layerLabels->getLabels())
        output << "\"" << label.text << "\" (" << label.id << "), ";
    }
    qCDebug(lcChan) << "distributeLabelsToLayers: " << output.str().c_str();
  }
}

void LabellingCoordinator::updateLabelPositionsInLabelNodes(
    LabelPositions labelPositions)
{
  hasChanges = false;
  for (auto &labelNode : nodes->getLabelNodes())
  {
    int labelId = labelNode->label.id;
    if (labelPositions.count(labelId))
    {
      auto newPosition = labelPositions.get3dFor(labelId);
      if (!hasChanges && (labelNode->labelPosition - newPosition).norm() > 1e-7)
      {
        hasChanges = true;
      }

      labelNode->setIsVisible(true);
      labelNode->labelPosition = labelPositions.get3dFor(labelId);
      labelNode->labelPositionNDC = labelPositions.getNDCFor(labelId);
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

void LabellingCoordinator::performInternalLabelling()
{
  for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
    labelsInLayer[layerIndex]->clear();

  LabelPositions labelPositions;
  auto container = labelsInLayer[0];
  for (auto label : labels->getLabels())
  {
    auto ndc = labellerFrameData.project(label.anchorPosition);
    labelPositions.update(label.id, ndc, label.anchorPosition);

    container->add(label);
    labelIdToLayerIndex[label.id] = 0;
    labelIdToZValue[label.id] = 1.0f;
  }

  updateLabelPositionsInLabelNodes(labelPositions);
}

