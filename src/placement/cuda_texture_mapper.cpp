#include "./cuda_texture_mapper.h"
#include <QtOpenGLExtensions>

CudaTextureMapper::CudaTextureMapper(unsigned int textureId, unsigned int flags)
  : textureId(textureId)
{
  qInfo() << "map texture" << textureId;
  HANDLE_ERROR(
      cudaGraphicsGLRegisterImage(&resource, textureId, GL_TEXTURE_2D, flags));
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

CudaTextureMapper *
CudaTextureMapper::createReadWriteMapper(unsigned int textureId)
{
  return new CudaTextureMapper(textureId,
                               cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

CudaTextureMapper *
CudaTextureMapper::createReadOnlyMapper(unsigned int textureId)
{
  return new CudaTextureMapper(textureId, cudaGraphicsRegisterFlagsReadOnly);
}

CudaTextureMapper *
CudaTextureMapper::createReadWriteDiscardMapper(unsigned int textureId)
{
  return new CudaTextureMapper(textureId,
                               cudaGraphicsRegisterFlagsWriteDiscard);
}

