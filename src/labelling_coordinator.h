#ifndef SRC_LABELLING_COORDINATOR_H_

#define SRC_LABELLING_COORDINATOR_H_

#include <Eigen/Core>
#include <memory>
#include <map>
#include <vector>
#include "./placement/labeller.h"
#include "./forces/labeller.h"
#include "./labelling/labeller_frame_data.h"
#include "./labelling/clustering.h"
#include "./graphics/buffer_drawer.h"

class PersistentConstraintUpdater;
class LabelsContainer;
class Labels;
class Nodes;
class TextureMapperManager;
namespace Placement
{
class OcclusionCalculator;
class IntegralCostsCalculator;
}

/**
 * \brief Coordinates all labelling related classes, i.e. the forces and
 * placement labellers
 *
 * Each frame #update must be called in the update phase, #updateClusters
 * before the HABuffer is used (so that the layer planes are up to date)
 * and #updatePlacement after everything has been rendered into the FBOs.
 */
class LabellingCoordinator
{
 public:
  LabellingCoordinator(int layerCount,
                       std::shared_ptr<Forces::Labeller> forcesLabeller,
                       std::shared_ptr<Labels> labels,
                       std::shared_ptr<Nodes> nodes);

  void initialize(int bufferSize,
                  std::shared_ptr<Graphics::BufferDrawer> drawer,
                  std::shared_ptr<TextureMapperManager> textureMapperManager,
                  int widht, int height);
  void cleanup();

  void update(double frameTime, Eigen::Matrix4f projection,
              Eigen::Matrix4f view, int activeLayerNumber = 0);
  void updatePlacement();
  std::vector<float> updateClusters();

  void resize(int width, int height);

  void saveOcclusion();
  void setCostFunctionWeights(Placement::CostFunctionWeights weights);

  bool forcesEnabled = true;

 private:
  int layerCount;
  std::shared_ptr<Forces::Labeller> forcesLabeller;
  std::shared_ptr<Labels> labels;
  std::shared_ptr<Nodes> nodes;

  std::shared_ptr<Placement::OcclusionCalculator> occlusionCalculator;
  std::shared_ptr<Placement::IntegralCostsCalculator> integralCostsCalculator;
  std::shared_ptr<PersistentConstraintUpdater> persistentConstraintUpdater;
  std::vector<std::shared_ptr<Placement::Labeller>> placementLabellers;
  std::vector<std::shared_ptr<LabelsContainer>> labelsInLayer;
  std::map<int, int> labelIdToLayerIndex;
  bool firstFramesWithoutPlacement = true;
  LabellerFrameData labellerFrameData;
  Clustering clustering;

  std::map<int, Eigen::Vector3f> getPlacementPositions(int activeLayerNumber);
  std::map<int, Eigen::Vector3f>
  getForcesPositions(std::map<int, Eigen::Vector3f> placementPositions);
  void distributeLabelsToLayers();
  void
  updateLabelPositionsInLabelNodes(std::map<int, Eigen::Vector3f> newPositions);
};

#endif  // SRC_LABELLING_COORDINATOR_H_
