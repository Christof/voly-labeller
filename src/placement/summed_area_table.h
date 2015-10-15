#ifndef SRC_PLACEMENT_SUMMED_AREA_TABLE_H_

#define SRC_PLACEMENT_SUMMED_AREA_TABLE_H_

#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

thrust::host_vector<float> algSAT(float *h_inout, int w, int h);

#endif  // SRC_PLACEMENT_SUMMED_AREA_TABLE_H_
