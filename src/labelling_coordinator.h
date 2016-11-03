#ifndef SRC_LABELLING_COORDINATOR_H_

#define SRC_LABELLING_COORDINATOR_H_

#include <Eigen/Core>
#include <memory>
#include <map>
#include <vector>
#include <limits>
#include "./placement/labeller.h"
#include "./forces/labeller.h"
#include "./labelling/labeller_frame_data.h"
#include "./labelling/clustering.h"
#include "./labelling/label_positions.h"
#include "./graphics/buffer_drawer.h"
#include "./utils/profiling_statistics.h"

class PersistentConstraintUpdater;
class LabelsContainer;
class Labels;
class Nodes;
class TextureMapperManager;
namespace Graphics
{
class Gl;
class Managers;
}
namespace Placement
{
class OcclusionCalculator;
class IntegralCostsCalculator;
class Saliency;
class LabelsArranger;
class ApolloniusLabelsArranger;
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

  void initialize(Graphics::Gl *gl, int bufferSize,
                  std::shared_ptr<Graphics::Managers> managers,
                  std::shared_ptr<TextureMapperManager> textureMapperManager,
                  int widht, int height);
  void cleanup();

  void setEnabled(bool enabled);

  /**
   * \brief Updates the label positions by calculating the placement results and
   * converting it into forces.
   *
   * @retval true if any position has changed
   * @retval false if no position has changed
   */
  bool update(double frameTime, bool isIdle, Eigen::Matrix4f projection,
              Eigen::Matrix4f view, int activeLayerNumber = 0);
  void updatePlacement();
  std::vector<float> updateClusters();

  bool haveLabelPositionsChanged();

  void resize(int width, int height);

  void saveOcclusion();
  void saveConstraints();
  void setCostFunctionWeights(Placement::CostFunctionWeights weights);

  bool forcesEnabled = true;
  bool optimizeOnIdle = false;
  bool useApollonius = false;
  bool hasChanges;

 private:
  int layerCount;
  std::shared_ptr<Forces::Labeller> forcesLabeller;
  std::shared_ptr<Labels> labels;
  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<Placement::LabelsArranger> insertionOrderLabelsArranger;
  std::shared_ptr<Placement::LabelsArranger> randomizedLabelsArranger;
  std::vector<std::shared_ptr<Placement::ApolloniusLabelsArranger>>
      apolloniusLabelsArrangers;

  std::shared_ptr<Placement::OcclusionCalculator> occlusionCalculator;
  std::shared_ptr<Placement::Saliency> saliency;
  std::shared_ptr<Placement::IntegralCostsCalculator> integralCostsCalculator;
  std::shared_ptr<PersistentConstraintUpdater> persistentConstraintUpdater;
  std::vector<std::shared_ptr<Placement::Labeller>> placementLabellers;
  std::vector<std::shared_ptr<LabelsContainer>> labelsInLayer;
  std::map<int, int> labelIdToLayerIndex;
  std::map<int, float> labelIdToZValue;
  bool isIdle;
  bool firstFramesWithoutPlacement = true;
  LabellerFrameData labellerFrameData;
  Clustering clustering;
  float sumOfCosts = std::numeric_limits<float>::max();
  bool preserveLastResult = false;
  bool labellingEnabled = true;
  std::map<int, Eigen::Vector2f> lastPlacementResult;

  std::map<int, Eigen::Vector2f> getPlacementPositions(int activeLayerNumber);
  LabelPositions getForcesPositions(LabelPositions placementPositions);
  void distributeLabelsToLayers();
  void updateLabelPositionsInLabelNodes(LabelPositions labelPositions);

  std::map<int, Eigen::Vector3f>
  addDepthValueNDC(std::map<int, Eigen::Vector2f> positionsNDC);
  std::map<int, Eigen::Vector3f>
  ndcPositionsTo3d(std::map<int, Eigen::Vector3f> positionsNDC);

  Eigen::Vector2f bufferSize;

  bool saveConstraintsInNextFrame = false;
  ProfilingStatistics profilingStatistics;
};

#endif  // SRC_LABELLING_COORDINATOR_H_
