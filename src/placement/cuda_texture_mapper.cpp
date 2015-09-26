#include "./cuda_texture_mapper.h"

cudaError registerImage(cudaGraphicsResource *resource, unsigned int id)
{
  return cudaGraphicsGLRegisterImage(&resource, id, GL_RENDERBUFFER,
                                     cudaGraphicsRegisterFlagsNone);
}

CudaTextureMapper::CudaTextureMapper(unsigned int textureId)
  : textureId(textureId)
{
  // cudaError error = registerImage(resource, textureId);
  qInfo() << "map texture" << textureId;
  cudaError error = cudaGraphicsGLRegisterImage(&resource, textureId, GL_TEXTURE_2D,
                                                cudaGraphicsRegisterFlagsNone);

  qWarning() << "register image" << cudaGetErrorString(error);
}

CudaTextureMapper::~CudaTextureMapper()
{
  cudaGraphicsUnregisterResource(resource);
}
