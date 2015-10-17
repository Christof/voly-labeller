#ifndef SRC_PLACEMENT_TO_GRAY_H_

#define SRC_PLACEMENT_TO_GRAY_H_

#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

/**
 * \brief
 *
 *
 */
class ToGray
{
 public:
  explicit ToGray(std::shared_ptr<CudaArrayProvider> imageProvider);

  void runKernel();

 private:
  std::shared_ptr<CudaArrayProvider> imageProvider;
  cudaSurfaceObject_t image;
};

#endif  // SRC_PLACEMENT_TO_GRAY_H_
