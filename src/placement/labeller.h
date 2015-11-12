#ifndef SRC_PLACEMENT_LABELLER_H_

#define SRC_PLACEMENT_LABELLER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include <map>
#include "../labelling/labels.h"
#include "../labelling/labeller_frame_data.h"
#include "./cost_function_calculator.h"

class SummedAreaTable;
class CudaArrayProvider;
class Apollonius;
class OccupancyUpdater;

namespace Placement
{

/**
 * \brief
 *
 *
 */
class Labeller
{
 public:
  explicit Labeller(std::shared_ptr<Labels> labels);

  void
  initialize(std::shared_ptr<CudaArrayProvider> occupancyTextureMapper,
             std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper,
             std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper);

  void setInsertionOrder(std::vector<int> ids);

  std::map<int, Eigen::Vector3f> update(const LabellerFrameData &frameData);

  std::map<int, Eigen::Vector3f> getLastPlacementResult();

  void resize(int width, int height);

  void cleanup();

 private:
  std::shared_ptr<Labels> labels;
  CostFunctionCalculator costFunctionCalculator;
  std::shared_ptr<Apollonius> apollonius;
  std::shared_ptr<SummedAreaTable> occupancySummedAreaTable;
  std::shared_ptr<OccupancyUpdater> occupancyUpdater;
  std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper;
  std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper;
  std::vector<int> insertionOrder;

  int width;
  int height;

  std::map<int, Eigen::Vector3f> newPositions;
};

}  // namespace Placement
#endif  // SRC_PLACEMENT_LABELLER_H_
