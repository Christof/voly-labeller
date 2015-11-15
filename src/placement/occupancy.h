#ifndef SRC_PLACEMENT_OCCUPANCY_H_

#define SRC_PLACEMENT_OCCUPANCY_H_

#include <cuda_runtime.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

/**
 * \brief Calculates occupancy from the given positions
 *
 * Currently only the depth is considered for the occupancy.
 * A depth value of 1 (background) yields an occupancy of 0. All values which are
 * not in the background result in a value larger 0.
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
