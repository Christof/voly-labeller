#ifndef SRC_PLACEMENT_OCCUPANCY_H_

#define SRC_PLACEMENT_OCCUPANCY_H_

#include <cuda_runtime.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

/**
 * \brief
 *
 *
 */
class Occupancy
{
 public:
  Occupancy(std::shared_ptr<CudaArrayProvider> positionProvider,
            std::shared_ptr<CudaArrayProvider> outputProvider);
  ~Occupancy();

  void runKernel();

 private:
  std::shared_ptr<CudaArrayProvider> positionProvider;
  std::shared_ptr<CudaArrayProvider> outputProvider;
  cudaTextureObject_t positions = 0;
  cudaSurfaceObject_t output = 0;

  void createSurfaceObjects();
};

#endif  // SRC_PLACEMENT_OCCUPANCY_H_
