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
 * \brief Maps a given OpenGl texture to a CUDA resource
 *
 * Per default to given texture is mapped so that CUDA
 * can read it and also write to it.
 * To use the texture #map must be called. Then CUDA
 * can access the texture memory via the cudaArray which
 * is returned by #getArray. After the changes have been
 * made #unmap must be invoked.
 */
class CudaTextureMapper : public CudaArrayProvider
{
 public:
  CudaTextureMapper(unsigned int textureId, int width, int height,
                    unsigned int flags);
  virtual ~CudaTextureMapper();

  virtual void map();
  virtual void unmap();

  virtual cudaChannelFormatDesc getChannelDesc();
  virtual cudaArray_t getArray();

  static CudaTextureMapper *createReadWriteMapper(unsigned int textureId,
                                                  int width, int height);
  static CudaTextureMapper *createReadOnlyMapper(unsigned int textureId,
                                                 int width, int height);
  static CudaTextureMapper *createReadWriteDiscardMapper(unsigned int textureId,
                                                         int width, int height);

 private:
  unsigned int textureId;
  unsigned int flags;
  cudaGraphicsResource *resource = 0;
  cudaChannelFormatDesc channelDesc;
  cudaArray_t array;
};

#endif  // SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_
