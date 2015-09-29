#ifndef SRC_PLACEMENT_SUMMED_AREA_TABLE_H_

#define SRC_PLACEMENT_SUMMED_AREA_TABLE_H_

#include <cuda_runtime.h>
#include <memory>
#include "./cuda_texture_mapper.h"

void toGray(std::shared_ptr<CudaTextureMapper> tex, int image_size);

#endif  // SRC_PLACEMENT_SUMMED_AREA_TABLE_H_
