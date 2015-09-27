#ifndef SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_

#define SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_

#include <QtOpenGLExtensions>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>

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

  cudaGraphicsResource **getResource();

 private:
  unsigned int textureId;
  cudaGraphicsResource *resource = 0;
};

#endif  // SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_
