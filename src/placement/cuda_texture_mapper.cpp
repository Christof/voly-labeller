#include "./cuda_texture_mapper.h"
#include <QtOpenGLExtensions>
#include <cuda.h>
#include <cudaGL.h>

CudaTextureMapper::CudaTextureMapper(unsigned int textureId)
  : textureId(textureId)
{
  CUresult error = cuGraphicsGLRegisterImage(
      &resource, textureId, GL_RENDERBUFFER, CU_GRAPHICS_REGISTER_FLAGS_NONE);
  qWarning() << "register image" << error;
}

CudaTextureMapper::~CudaTextureMapper()
{
  cuGraphicsUnregisterResource(resource);
}
