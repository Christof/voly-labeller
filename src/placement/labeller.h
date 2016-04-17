#ifndef SRC_PLACEMENT_LABELLER_H_

#define SRC_PLACEMENT_LABELLER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include <map>
#include "../labelling/labels_container.h"
#include "../labelling/labeller_frame_data.h"
#include "./cost_function_calculator.h"

class CudaArrayProvider;
class ConstraintUpdater;
class PersistentConstraintUpdater;

/**
 * \brief Contains classes for label placement using a global minimization of a
 * cost function
 *
 */
namespace Placement
{

class Apollonius;
class OccupancyUpdater;
class SummedAreaTable;

/**
 * \brief Places labels according to an energy minimization of a cost function
 *
 * The label positions are updated and retrieved with #update.
 */
class Labeller
{
 public:
  explicit Labeller(std::shared_ptr<LabelsContainer> labels);

  void
  initialize(std::shared_ptr<CudaArrayProvider> integralCostsTextureMapper,
             std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper,
             std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper,
             std::shared_ptr<CudaArrayProvider> constraintTextureMapper,
             std::shared_ptr<PersistentConstraintUpdater> constraintUpdater);

  std::map<int, Eigen::Vector3f> update(const LabellerFrameData &frameData);

  std::map<int, Eigen::Vector3f> getLastPlacementResult();

  void resize(int width, int height);

  void cleanup();

  bool useApollonius = false;

 private:
  std::shared_ptr<LabelsContainer> labels;
  std::unique_ptr<CostFunctionCalculator> costFunctionCalculator;
  std::shared_ptr<Apollonius> apollonius;
  std::shared_ptr<SummedAreaTable> integralCosts;
  std::shared_ptr<OccupancyUpdater> occupancyUpdater;
  std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper;
  std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper;
  std::shared_ptr<PersistentConstraintUpdater> constraintUpdater;

  Eigen::Vector2i size;

  std::map<int, Eigen::Vector3f> newPositions;

  std::vector<Eigen::Vector4f> createLabelSeeds(Eigen::Vector2i size,
                                                Eigen::Matrix4f viewProjection);
  std::vector<Label>
  getLabelsInApolloniusOrder(const LabellerFrameData &frameData,
                             Eigen::Vector2i bufferSize);
  Eigen::Vector3f reprojectTo3d(Eigen::Vector2i newPosition, float anchorZValue,
                                Eigen::Vector2i bufferSize,
                                Eigen::Matrix4f inverseViewProjection);
  Eigen::Vector2f toPixel(Eigen::Vector3f ndc, Eigen::Vector2i size);
};

}  // namespace Placement
#endif  // SRC_PLACEMENT_LABELLER_H_
