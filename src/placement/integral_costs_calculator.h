#ifndef SRC_PLACEMENT_INTEGRAL_COSTS_CALCULATOR_H_

#define SRC_PLACEMENT_INTEGRAL_COSTS_CALCULATOR_H_

#include <memory>
#include "../utils/cuda_array_provider.h"

namespace Placement
{

/**
 * \brief
 *
 *
 */
class IntegralCostsCalculator
{
 public:
  IntegralCostsCalculator(
      std::shared_ptr<CudaArrayProvider> occlusionProvider,
      std::shared_ptr<CudaArrayProvider> outputProvider);
  ~IntegralCostsCalculator();

  void runKernel();

 private:
  std::shared_ptr<CudaArrayProvider> occlusionProvider;
  std::shared_ptr<CudaArrayProvider> outputProvider;

  cudaTextureObject_t occlusion = 0;
  cudaSurfaceObject_t output = 0;

  void createSurfaceObjects();
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_INTEGRAL_COSTS_CALCULATOR_H_
