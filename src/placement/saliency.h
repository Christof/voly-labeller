#ifndef SRC_PLACEMENT_SALIENCY_H_

#define SRC_PLACEMENT_SALIENCY_H_

#include <cuda_runtime.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

namespace Placement
{

/**
 * \brief
 *
 *
 */
class Saliency
{
 public:
  Saliency(std::shared_ptr<CudaArrayProvider> inputProvider,
           std::shared_ptr<CudaArrayProvider> outputProvider);
  virtual ~Saliency();

  void runKernel();

 private:
  std::shared_ptr<CudaArrayProvider> inputProvider;
  std::shared_ptr<CudaArrayProvider> outputProvider;
  cudaTextureObject_t input = 0;
  cudaSurfaceObject_t output = 0;

  void createSurfaceObjects();
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_SALIENCY_H_
