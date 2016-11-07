#ifndef SRC_PLACEMENT_DIRECT_INTEGRAL_COSTS_CALCULATOR_H_

#define SRC_PLACEMENT_DIRECT_INTEGRAL_COSTS_CALCULATOR_H_

#include <memory>
#include "../utils/cuda_array_provider.h"
#include "./integral_costs_calculator.h"

namespace Placement
{

/**
 * \brief
 *
 *
 */
class DirectIntegralCostsCalculator
{
 public:
  DirectIntegralCostsCalculator(
      std::shared_ptr<CudaArrayProvider> colorProvider,
      std::shared_ptr<CudaArrayProvider> saliencyProvider,
      std::shared_ptr<CudaArrayProvider> outputProvider);
  virtual ~DirectIntegralCostsCalculator();

  void runKernel(int layerIndex, int layerCount);

  IntegralCostsWeights weights;

 private:
  std::shared_ptr<CudaArrayProvider> colorProvider;
  std::shared_ptr<CudaArrayProvider> saliencyProvider;
  std::shared_ptr<CudaArrayProvider> outputProvider;

  cudaTextureObject_t color = 0;
  cudaTextureObject_t saliency = 0;
  cudaSurfaceObject_t output = 0;

  void createSurfaceObjects();
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_DIRECT_INTEGRAL_COSTS_CALCULATOR_H_
