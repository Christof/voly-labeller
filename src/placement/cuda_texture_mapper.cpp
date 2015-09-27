#include "./cuda_texture_mapper.h"

CudaTextureMapper::CudaTextureMapper(unsigned int textureId)
  : textureId(textureId)
{
  qInfo() << "map texture" << textureId;
  cudaError error = cudaGraphicsGLRegisterImage(
      &resource, textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

  qWarning() << "register image" << cudaGetErrorString(error);
}

CudaTextureMapper::~CudaTextureMapper()
{
  cudaGraphicsUnregisterResource(resource);
}

cudaGraphicsResource **CudaTextureMapper::getResource()
{
  return &resource;
}
