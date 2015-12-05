#ifndef SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_

#define SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_

#include <thrust/device_vector.h>
#include <tuple>
#include "../utils/cuda_array_provider.h"

namespace Placement
{

/**
 * \brief
 *
 *
 */
class CostFunctionCalculator
{
 public:
  explicit CostFunctionCalculator(
      std::shared_ptr<CudaArrayProvider> constraintImage);
  ~CostFunctionCalculator();

  void resize(int width, int height);
  void setTextureSize(int width, int height);

  void calculateCosts(const thrust::device_vector<float> &distances);
  std::tuple<float, float> calculateForLabel(
      const thrust::device_vector<float> &occupancySummedAreaTable, int labelId,
      float anchorX, float anchorY, int labelWidthInPixel,
      int labelHeightInPixel);

 private:
  int width;
  int height;

  int textureWidth;
  int textureHeight;

  std::shared_ptr<CudaArrayProvider> constraintImage;
  cudaTextureObject_t constraints = 0;

  void createTextureObject();
};

}  // namespace Placement
#endif  // SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_
