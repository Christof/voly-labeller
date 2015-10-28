#ifndef SRC_PLACEMENT_LABELLER_H_

#define SRC_PLACEMENT_LABELLER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include "../labelling/labels.h"
#include "../labelling/labeller_frame_data.h"
#include "./cost_function_calculator.h"

class SummedAreaTable;
class CudaArrayProvider;

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

  void initialize(std::shared_ptr<CudaArrayProvider> occupancyTextureMapper);

  void setInsertionOrder(std::vector<int> ids);

  std::map<int, Eigen::Vector3f> update(const LabellerFrameData &frameData);

  void resize(int width, int height);

  void cleanup();

 private:
  std::shared_ptr<Labels> labels;
  CostFunctionCalculator costFunctionCalculator;
  std::shared_ptr<SummedAreaTable> occupancySummedAreaTable;
  std::vector<int> insertionOrder;

  int width;
  int height;
};

}  // namespace Placement
#endif  // SRC_PLACEMENT_LABELLER_H_
