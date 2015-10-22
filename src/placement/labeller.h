#ifndef SRC_PLACEMENT_LABELLER_H_

#define SRC_PLACEMENT_LABELLER_H_

#include <Eigen/Core>
#include <memory>
#include "../labelling/labels.h"
#include "../labelling/labeller_frame_data.h"
#include "./cost_function_calculator.h"

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

  void initialize(std::shared_ptr<CudaArrayProvider> occupancySummedAreaTable);

  std::map<int, Eigen::Vector3f> update(const LabellerFrameData &frameData);

 private:
  std::shared_ptr<Labels> labels;
  CostFunctionCalculator costFunctionCalculator;
  std::shared_ptr<CudaArrayProvider> occupancySummedAreaTable;
};

}  // namespace Placement
#endif  // SRC_PLACEMENT_LABELLER_H_
