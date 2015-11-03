#include "./cuda_texture_mapper.h"
#include <QtOpenGLExtensions>

CudaTextureMapper::CudaTextureMapper(unsigned int textureId, int width,
                                     int height, unsigned int flags)
  : CudaArrayProvider(width, height), textureId(textureId), flags(flags)
{
}

CudaTextureMapper::~CudaTextureMapper()
{
  if (resource)
    HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
}

void CudaTextureMapper::map()
{
  if (!resource)
  {
    qInfo() << "map texture" << textureId;
    HANDLE_ERROR(cudaGraphicsGLRegisterImage(&resource, textureId,
                                             GL_TEXTURE_2D, flags));
  }

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

CudaTextureMapper *
CudaTextureMapper::createReadWriteMapper(unsigned int textureId, int width,
                                         int height)
{
  return new CudaTextureMapper(textureId, width, height,
                               cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

CudaTextureMapper *
CudaTextureMapper::createReadOnlyMapper(unsigned int textureId, int width,
                                        int height)
{
  return new CudaTextureMapper(textureId, width, height,
                               cudaGraphicsRegisterFlagsReadOnly);
}

CudaTextureMapper *
CudaTextureMapper::createReadWriteDiscardMapper(unsigned int textureId,
                                                int width, int height)
{
  return new CudaTextureMapper(textureId, width, height,
                               cudaGraphicsRegisterFlagsWriteDiscard);
}

