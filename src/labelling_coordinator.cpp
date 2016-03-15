#include "./labelling_coordinator.h"
#include <Eigen/Core>
#include <map>
#include <vector>
#include "./labelling/labels_container.h"
#include "./labelling/clustering.h"
#include "./labelling/labels.h"
#include "./placement/constraint_updater.h"
#include "./placement/persistent_constraint_updater.h"
#include "./placement/cuda_texture_mapper.h"
#include "./graphics/buffer_drawer.h"
#include "./nodes.h"
#include "./label_node.h"
#include "./texture_mapper_manager.h"

LabellingCoordinator::LabellingCoordinator(
    int layerCount, std::shared_ptr<Forces::Labeller> forcesLabeller,
    std::shared_ptr<Labels> labels, std::shared_ptr<Nodes> nodes)
  : layerCount(layerCount), forcesLabeller(forcesLabeller), labels(labels),
    nodes(nodes), clustering(labels, layerCount - 1)
{
}

void LabellingCoordinator::initialize(
    int bufferSize, std::shared_ptr<Graphics::BufferDrawer> drawer,
    std::shared_ptr<TextureMapperManager> textureMapperManager, int width,
    int height)
{
  auto constraintUpdater =
      std::make_shared<ConstraintUpdater>(drawer, bufferSize, bufferSize);
  persistentConstraintUpdater =
      std::make_shared<PersistentConstraintUpdater>(constraintUpdater);

  for (int layerIndex = 0; layerIndex < layerCount; ++layerIndex)
  {
    auto labelsContainer = std::make_shared<LabelsContainer>();
    labelsInLayer.push_back(labelsContainer);
    auto labeller = std::make_shared<Placement::Labeller>(labelsContainer);
    labeller->resize(width, height);
    labeller->initialize(
        textureMapperManager->getOccupancyTextureMapper(layerIndex),
        textureMapperManager->getDistanceTransformTextureMapper(layerIndex),
        textureMapperManager->getApolloniusTextureMapper(layerIndex),
        textureMapperManager->getConstraintTextureMapper(),
        persistentConstraintUpdater);

    placementLabellers.push_back(labeller);
  }
}

void LabellingCoordinator::cleanup()
{
  for (auto placementLabeller : placementLabellers)
    placementLabeller->cleanup();
}

void LabellingCoordinator::update(double frameTime, Eigen::Matrix4f projection,
                                  Eigen::Matrix4f view, int activeLayerNumber)
{
  persistentConstraintUpdater->clear();
  std::map<int, Eigen::Vector3f> placementPositions;
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

  labellerFrameData = LabellerFrameData(frameTime, projection, view);
  if (firstFramesWithoutPlacement && placementPositions.size())
  {
    firstFramesWithoutPlacement = false;
    forcesLabeller->setPositions(labellerFrameData, placementPositions);
  }

  auto newPositions =
      forcesLabeller->update(labellerFrameData, placementPositions);

  for (auto &labelNode : nodes->getLabelNodes())
  {
    labelNode->labelPosition = newPositions[labelNode->label.id];
  }

  auto centerWithLabelIds = clustering.getCentersWithLabelIds();
  layerIndex = 0;
  for (auto &pair : centerWithLabelIds)
  {
    auto &container = labelsInLayer[layerIndex];
    container->clear();

    for (int labelId : pair.second)
      container->add(labels->getById(labelId));

    layerIndex++;
  }

  for (auto &labelNode : nodes->getLabelNodes())
  {
    if (newPositions.count(labelNode->label.id))
    {
      labelNode->setIsVisible(true);
      labelNode->labelPosition = newPositions[labelNode->label.id];
    }
    else
    {
      labelNode->setIsVisible(false);
    }
  }
}

void LabellingCoordinator::updatePlacement()
{
  for (auto placementLabeller : placementLabellers)
    placementLabeller->update(labellerFrameData);
}

std::vector<float> LabellingCoordinator::updateClusters()
{
  clustering.update(labellerFrameData.viewProjection);
  auto clusters = clustering.getFarthestClusterMembersWithLabelIds();
  std::vector<float> zValues;

  std::cout << "zValuesEye: ";
  for (auto pair : clusters)
  {
    zValues.push_back(pair.first);
    std::cout << pair.first << ":";
    for (auto z : pair.second)
      std::cout << z << ", ";
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return zValues;
}

void LabellingCoordinator::resize(int width, int height)
{
  for (auto placementLabeller : placementLabellers)
    placementLabeller->resize(width, height);

  forcesLabeller->resize(width, height);
}
