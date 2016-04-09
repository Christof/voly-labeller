#ifndef SRC_PLACEMENT_OCCLUSION_H_

#define SRC_PLACEMENT_OCCLUSION_H_

#include <cuda_runtime.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

namespace Placement
{

/**
 * \brief Calculate the occlusion by using the alpha value of the color image
 *
 * If the output is smaller than the input, the smallest transparency (highest
 * alpha value) of the sampled region is used.
 *
 * The value can either be set directly to the output via #calculateOcclusion,
 * or added to the current value of the output using #addOcclusion. The later
 * is used to sum up the occlusions of the layers.
 */
class Occlusion
{
 public:
  Occlusion(std::shared_ptr<CudaArrayProvider> colorProvider,
            std::shared_ptr<CudaArrayProvider> outputProvider);
  ~Occlusion();

  void addOcclusion();
  void calculateOcclusion();

 private:
  void runKernel(bool addToOutputValue);
  std::shared_ptr<CudaArrayProvider> colorProvider;
  std::shared_ptr<CudaArrayProvider> outputProvider;
  cudaTextureObject_t positions = 0;
  cudaSurfaceObject_t output = 0;

  void createSurfaceObjects();
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_OCCLUSION_H_
