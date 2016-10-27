#ifndef SRC_PLACEMENT_CUDA_TEXTURE_3D_MAPPER_H_

#define SRC_PLACEMENT_CUDA_TEXTURE_3D_MAPPER_H_

typedef unsigned int GLuint;
typedef unsigned int GLenum;
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include "../utils/cuda_helper.h"
#include "../utils/cuda_array_provider.h"

/**
 * \brief Maps a given OpenGl 3d texture to a CUDA resource
 *
 * Per default to given texture is mapped so that CUDA
 * can read it and also write to it.
 * To use the texture #map must be called. Then CUDA
 * can access the texture memory via the cudaArray which
 * is returned by #getArray. After the changes have been
 * made #unmap must be invoked.
 */
class CudaTexture3DMapper : public CudaArrayProvider
{
 public:
  CudaTexture3DMapper(unsigned int textureId, int width, int height, int depth,
                      unsigned int flags);
  virtual ~CudaTexture3DMapper();

  virtual void map();
  virtual void unmap();

  virtual cudaChannelFormatDesc getChannelDesc();
  virtual cudaArray_t getArray();

  static CudaTexture3DMapper *createReadWriteMapper(unsigned int textureId,
                                                    int width, int height,
                                                    int depth);
  static CudaTexture3DMapper *createReadOnlyMapper(unsigned int textureId,
                                                   int width, int height,
                                                   int depth);
  static CudaTexture3DMapper *
  createReadWriteDiscardMapper(unsigned int textureId, int width, int height,
                               int depth);

 private:
  unsigned int textureId;
  unsigned int flags;
  cudaGraphicsResource *resource = 0;
  cudaChannelFormatDesc channelDesc;
  cudaArray_t array;
};

#endif  // SRC_PLACEMENT_CUDA_TEXTURE_3D_MAPPER_H_
