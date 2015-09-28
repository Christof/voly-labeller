#include "./cuda_texture_mapper.h"
#include "../utils/cuda_helper.h"

CudaTextureMapper::CudaTextureMapper(unsigned int textureId)
  : textureId(textureId)
{
  qInfo() << "map texture" << textureId;
  HANDLE_ERROR(cudaGraphicsGLRegisterImage(
      &resource, textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
}

CudaTextureMapper::~CudaTextureMapper()
{
  HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
}

cudaGraphicsResource **CudaTextureMapper::getResource()
{
  return &resource;
}
