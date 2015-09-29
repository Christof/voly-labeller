#ifndef SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_

#define SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_

typedef unsigned int GLuint;
typedef unsigned int GLenum;
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include "../utils/cuda_helper.h"

/**
 * \brief
 *
 *
 */
class CudaTextureMapper
{
 public:
  CudaTextureMapper(unsigned int textureId);
  virtual ~CudaTextureMapper();

  void map();
  void unmap();

  cudaGraphicsResource **getResource();
  cudaChannelFormatDesc getChannelDesc();
  cudaArray_t getArray();

 private:
  unsigned int textureId;
  cudaGraphicsResource *resource = 0;
  cudaChannelFormatDesc channelDesc;
  cudaArray_t array;
};

#endif  // SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_
