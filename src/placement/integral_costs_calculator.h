#ifndef SRC_PLACEMENT_INTEGRAL_COSTS_CALCULATOR_H_

#define SRC_PLACEMENT_INTEGRAL_COSTS_CALCULATOR_H_

#include <memory>
#include "../utils/cuda_array_provider.h"

namespace Placement
{

/**
 * \brief Computes the weighted sum of the integral costs
 *
 * The integral costs constist of:
 * - occlusion
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

  float occlusionWeight = 1.0f;
  float saliencyWeight = 1e-3f;


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
