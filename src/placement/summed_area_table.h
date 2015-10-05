#ifndef SRC_PLACEMENT_SUMMED_AREA_TABLE_H_

#define SRC_PLACEMENT_SUMMED_AREA_TABLE_H_

#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <memory>
#include "./cuda_texture_mapper.h"

thrust::host_vector<float> algSAT(float *h_inout, int w, int h);

void toGray(std::shared_ptr<CudaTextureMapper> tex, int image_size);

#endif  // SRC_PLACEMENT_SUMMED_AREA_TABLE_H_
