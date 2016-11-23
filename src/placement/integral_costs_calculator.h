#ifndef SRC_PLACEMENT_INTEGRAL_COSTS_CALCULATOR_H_

#define SRC_PLACEMENT_INTEGRAL_COSTS_CALCULATOR_H_

#include <memory>
#include "./integral_costs_weights.h"
#include "../utils/cuda_array_provider.h"

namespace Placement
{

/**
 * \brief Computes the weighted sum of the integral costs
 *
 * The integral costs constist of:
 * - occlusion
 * - saliency
 *
 * These values are combined as weighted sum.
 */
class IntegralCostsCalculator
{
 public:
  IntegralCostsCalculator(
      std::shared_ptr<CudaArrayProvider> occlusionProvider,
      std::shared_ptr<CudaArrayProvider> saliencyProvider,
      std::shared_ptr<CudaArrayProvider> outputProvider);
  ~IntegralCostsCalculator();

  void runKernel();

  IntegralCostsWeights weights;

 private:
  std::shared_ptr<CudaArrayProvider> occlusionProvider;
  std::shared_ptr<CudaArrayProvider> saliencyProvider;
  std::shared_ptr<CudaArrayProvider> outputProvider;

  cudaTextureObject_t occlusion = 0;
  cudaTextureObject_t saliency = 0;
  cudaSurfaceObject_t output = 0;

  void createSurfaceObjects();
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_INTEGRAL_COSTS_CALCULATOR_H_
