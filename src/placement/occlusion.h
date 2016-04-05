#ifndef SRC_PLACEMENT_OCCLUSION_H_

#define SRC_PLACEMENT_OCCLUSION_H_

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "../utils/cuda_array_provider.h"

namespace Placement
{

class Occlusion
{
 public:
  Occlusion(std::vector<std::shared_ptr<CudaArrayProvider>> colorProviders,
            std::shared_ptr<CudaArrayProvider> outputProvider);
  ~Occlusion();

  void runKernel();

 private:
  std::vector<std::shared_ptr<CudaArrayProvider>> colorProviders;
  std::shared_ptr<CudaArrayProvider> outputProvider;
  cudaTextureObject_t positions = 0;
  cudaSurfaceObject_t output = 0;

  void createSurfaceObjects();
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_OCCLUSION_H_