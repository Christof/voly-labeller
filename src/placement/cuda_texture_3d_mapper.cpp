#include <QtOpenGLExtensions>
#include "./cuda_texture_3d_mapper.h"

CudaTexture3DMapper::CudaTexture3DMapper(unsigned int textureId, int width,
                                     int height, int depth, unsigned int flags)
  : CudaArrayProvider(width, height), textureId(textureId), flags(flags)
{
  this->depth = depth;
}

CudaTexture3DMapper::~CudaTexture3DMapper()
{
  if (resource)
    HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
}

void CudaTexture3DMapper::map()
{
  if (!resource)
  {
    qInfo() << "map texture" << textureId;
    HANDLE_ERROR(cudaGraphicsGLRegisterImage(&resource, textureId,
                                             GL_TEXTURE_3D, flags));
  }

  HANDLE_ERROR(cudaGraphicsMapResources(1, &resource));
  HANDLE_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
  HANDLE_ERROR(cudaGetChannelDesc(&channelDesc, array));
}

void CudaTexture3DMapper::unmap()
{
  HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource));
}

cudaChannelFormatDesc CudaTexture3DMapper::getChannelDesc()
{
  return channelDesc;
}

cudaArray_t CudaTexture3DMapper::getArray()
{
  return array;
}

CudaTexture3DMapper *
CudaTexture3DMapper::createReadWriteMapper(unsigned int textureId, int width,
                                         int height, int depth)
{
  return new CudaTexture3DMapper(textureId, width, height, depth,
                               cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

CudaTexture3DMapper *
CudaTexture3DMapper::createReadOnlyMapper(unsigned int textureId, int width,
                                        int height, int depth)
{
  return new CudaTexture3DMapper(textureId, width, height, depth,
                               cudaGraphicsRegisterFlagsReadOnly);
}

CudaTexture3DMapper *CudaTexture3DMapper::createReadWriteDiscardMapper(
    unsigned int textureId, int width, int height, int depth)
{
  return new CudaTexture3DMapper(textureId, width, height, depth,
                               cudaGraphicsRegisterFlagsWriteDiscard);
}

