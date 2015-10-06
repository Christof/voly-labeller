#include "./cuda_texture_mapper.h"
#include <QtOpenGLExtensions>

CudaTextureMapper::CudaTextureMapper(unsigned int textureId)
  : textureId(textureId)
{
  qInfo() << "map texture" << textureId;
  HANDLE_ERROR(
      cudaGraphicsGLRegisterImage(&resource, textureId, GL_TEXTURE_2D,
                                  cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

CudaTextureMapper::~CudaTextureMapper()
{
  HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
}

void CudaTextureMapper::map()
{
  HANDLE_ERROR(cudaGraphicsMapResources(1, &resource));
  HANDLE_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
  HANDLE_ERROR(cudaGetChannelDesc(&channelDesc, array));
}

void CudaTextureMapper::unmap()
{
  HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource));
}

cudaChannelFormatDesc CudaTextureMapper::getChannelDesc()
{
  return channelDesc;
}

cudaArray_t CudaTextureMapper::getArray()
{
  return array;
}

