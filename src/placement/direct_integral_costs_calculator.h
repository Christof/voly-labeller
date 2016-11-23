#ifndef SRC_PLACEMENT_DIRECT_INTEGRAL_COSTS_CALCULATOR_H_

#define SRC_PLACEMENT_DIRECT_INTEGRAL_COSTS_CALCULATOR_H_

#include <memory>
#include "../utils/cuda_array_provider.h"
#include "./integral_costs_weights.h"

namespace Placement
{

/**
 * \brief Computes the integral costs directly from layer colors and saliency
 *
 * The occlusion is calculated on the fly from the layer colors. The occlusion
 * is partly weighted by the saliency as controled by
 * IntegralCostsWeights#fixOcclusionPart \f$\beta\f$:
 *
 * \f$\sum_{i=0}^l \texttt{alpha}_i + [(1 - \beta) \cdot
 * \texttt{saliencyValue} + \beta] \cdot
 * \sum_{i=l+1}^{n-1} \texttt{alpha}_i\f$.
 *
 * Where \f$l\f$ is the current layer index, \f$n\f$ is the number of layers
 * and \f$\texttt{alpha}\f$ is the alpha value from layer color.
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
