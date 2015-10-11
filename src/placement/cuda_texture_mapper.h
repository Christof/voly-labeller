#ifndef SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_

#define SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_

typedef unsigned int GLuint;
typedef unsigned int GLenum;
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include "../utils/cuda_helper.h"
#include "../utils/cuda_array_provider.h"

/**
 * \brief
 *
 *
 */
class CudaTextureMapper : public CudaArrayProvider
{
 public:
  explicit CudaTextureMapper(unsigned int textureId, unsigned int flags);
  virtual ~CudaTextureMapper();

  virtual void map();
  virtual void unmap();

  virtual cudaChannelFormatDesc getChannelDesc();
  virtual cudaArray_t getArray();

  static CudaTextureMapper* createReadWriteMapper(unsigned int textureId);
  static CudaTextureMapper* createReadOnlyMapper(unsigned int textureId);
  static CudaTextureMapper* createReadWriteDiscardMapper(unsigned int textureId);
 private:
  unsigned int textureId;
  cudaGraphicsResource *resource = 0;
  cudaChannelFormatDesc channelDesc;
  cudaArray_t array;
};

#endif  // SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_
