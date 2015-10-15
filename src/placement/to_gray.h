#ifndef SRC_PLACEMENT_TO_GRAY_H_

#define SRC_PLACEMENT_TO_GRAY_H_

#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <memory>
#include "../utils/cuda_array_provider.h"


void toGray(std::shared_ptr<CudaArrayProvider> tex, int image_size);

#endif  // SRC_PLACEMENT_TO_GRAY_H_
