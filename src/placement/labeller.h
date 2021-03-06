#ifndef SRC_PLACEMENT_LABELLER_H_

#define SRC_PLACEMENT_LABELLER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include <map>
#include "../labelling/labels_container.h"
#include "../labelling/labeller_frame_data.h"
#include "../labelling/label_positions.h"
#include "./cost_function_calculator.h"

class CudaArrayProvider;
class ConstraintUpdater;
class PersistentConstraintUpdater;

namespace Placement
{

class SummedAreaTable;
class LabelsArranger;

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
             std::shared_ptr<CudaArrayProvider> constraintTextureMapper,
             std::shared_ptr<PersistentConstraintUpdater> constraintUpdater);

  /**
   * \brief Calculates new label positions via global minimization and returns
   * them as NDC coordinates.
   */
  std::map<int, Eigen::Vector2f>
  update(const LabellerFrameData &frameData, bool ignoreOldPosition,
         const LabelPositions &oldLabelPositions);

  std::map<int, Eigen::Vector2f> getLastPlacementResult();
  float getLastSumOfCosts();

  void resize(int width, int height);

  void cleanup();

  void setCostFunctionWeights(CostFunctionWeights weights);

  void setLabelsArranger(std::shared_ptr<LabelsArranger> labelsArranger);

 private:
  std::shared_ptr<LabelsContainer> labels;
  std::unique_ptr<CostFunctionCalculator> costFunctionCalculator;
  std::shared_ptr<SummedAreaTable> integralCosts;
  std::shared_ptr<PersistentConstraintUpdater> constraintUpdater;
  std::shared_ptr<LabelsArranger> labelsArranger;

  Eigen::Vector2i size;
  Eigen::Vector2i bufferSize;

  std::map<int, Eigen::Vector2f> newPositions;
  std::map<int, Eigen::Vector2f> oldPositions;
  Eigen::Matrix4f oldViewProjectionMatrix;
  float costSum = 0.0f;
};

}  // namespace Placement
#endif  // SRC_PLACEMENT_LABELLER_H_
